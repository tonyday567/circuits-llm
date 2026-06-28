{-# LANGUAGE OverloadedStrings, BangPatterns #-}

-- | Full backward pass through one transformer block.
--   This is the core of the training backprop chain.
module Circuit.LLM.Backprop
  ( -- * Full backward pass
    gptBackward

    -- * Gradient types
  , GptGrads (..)
  , BlockGrads (..)
  , zeroGptGrads
  , addGptGrads
  , scaleGptGrads

    -- * Primitives (exported for testing)
  , linearBwd
  , layerNormBwd
  , geluBwd
  , softmaxBwd
  , crossEntropyBwd
  ) where

import Circuit.LLM.GPT
  ( FeedForward (..), Gpt (..), GptConfig (..), TransformerBlock (..), forward )
import Data.Foldable (foldl')
import Data.List (foldl1')
import Numeric.LinearAlgebra
  ( Matrix, Vector, cmap, fromList, fromLists, fromRows, maxElement, reshape
  , rows, cols, scale, sumElements, toList, toRows, tr, (|||)
  )
import qualified Numeric.LinearAlgebra as LA

----------------------------------------------------------------------
-- Gradient accumulators (unchanged)
----------------------------------------------------------------------

data GptGrads = GptGrads
  { ggWte  :: !(Matrix Double), ggWpe  :: !(Matrix Double)
  , ggBlocks :: ![BlockGrads]
  , ggLnGamma :: !(Vector Double), ggLnBeta  :: !(Vector Double)
  , ggHead   :: !(Matrix Double), ggHeadB  :: !(Vector Double)
  }

data BlockGrads = BlockGrads
  { bgAttnWq, bgAttnWk, bgAttnWv, bgAttnWo :: !(Matrix Double)
  , bgAttnLnGamma, bgAttnLnBeta :: !(Vector Double)
  , bgFfnW1, bgFfnW2 :: !(Matrix Double)
  , bgFfnB1, bgFfnB2 :: !(Vector Double)
  , bgFfnLnGamma, bgFfnLnBeta :: !(Vector Double)
  }

zeroGptGrads :: GptConfig -> GptGrads
zeroGptGrads cfg =
  let nEmb = gptNEmbd cfg; nLayer = gptNLayer cfg
      vocab = gptVocabSize cfg; ffMul = 4
      zM r c = reshape c (fromList (replicate (r * c) 0))
      zV n = fromList (replicate n 0)
  in  GptGrads
        { ggWte = zM vocab nEmb, ggWpe = zM 1024 nEmb
        , ggBlocks = replicate nLayer $ BlockGrads
            { bgAttnWq = zM nEmb nEmb, bgAttnWk = zM nEmb nEmb
            , bgAttnWv = zM nEmb nEmb, bgAttnWo = zM nEmb nEmb
            , bgAttnLnGamma = zV nEmb, bgAttnLnBeta = zV nEmb
            , bgFfnW1 = zM nEmb (ffMul * nEmb), bgFfnB1 = zV (ffMul * nEmb)
            , bgFfnW2 = zM (ffMul * nEmb) nEmb, bgFfnB2 = zV nEmb
            , bgFfnLnGamma = zV nEmb, bgFfnLnBeta = zV nEmb
            }
        , ggLnGamma = zV nEmb, ggLnBeta = zV nEmb
        , ggHead = zM nEmb vocab, ggHeadB = zV vocab
        }

addGptGrads :: GptGrads -> GptGrads -> GptGrads
addGptGrads a b =
  let r1 = rows (ggWte a); c1 = cols (ggWte a)
      r2 = rows (ggWte b); c2 = cols (ggWte b)
  in  if r1 /= r2 || c1 /= c2
        then error $ "addGptGrads Wte mismatch: (" ++ show r1 ++ "," ++ show c1 ++ ") vs (" ++ show r2 ++ "," ++ show c2 ++ ")"
        else GptGrads
          { ggWte = ggWte a + ggWte b, ggWpe = ggWpe a + ggWpe b
          , ggBlocks = zipWith addBG (ggBlocks a) (ggBlocks b)
          , ggLnGamma = ggLnGamma a + ggLnGamma b, ggLnBeta = ggLnBeta a + ggLnBeta b
          , ggHead = ggHead a + ggHead b, ggHeadB = ggHeadB a + ggHeadB b
          }
  where addBG x y = 
          let r1 = rows (bgAttnWq x); c1 = cols (bgAttnWq x)
              r2 = rows (bgAttnWq y); c2 = cols (bgAttnWq y)
          in  if r1 /= r2 || c1 /= c2
                then error $ "addBG Wq mismatch: (" ++ show r1 ++ "," ++ show c1 ++ ") vs (" ++ show r2 ++ "," ++ show c2 ++ ")"
                else BlockGrads
                  { bgAttnWq = bgAttnWq x + bgAttnWq y, bgAttnWk = bgAttnWk x + bgAttnWk y
                  , bgAttnWv = bgAttnWv x + bgAttnWv y, bgAttnWo = bgAttnWo x + bgAttnWo y
                  , bgAttnLnGamma = bgAttnLnGamma x + bgAttnLnGamma y
                  , bgAttnLnBeta  = bgAttnLnBeta x + bgAttnLnBeta y
                  , bgFfnW1 = bgFfnW1 x + bgFfnW1 y, bgFfnB1 = bgFfnB1 x + bgFfnB1 y
                  , bgFfnW2 = bgFfnW2 x + bgFfnW2 y, bgFfnB2 = bgFfnB2 x + bgFfnB2 y
                  , bgFfnLnGamma = bgFfnLnGamma x + bgFfnLnGamma y
                  , bgFfnLnBeta  = bgFfnLnBeta x + bgFfnLnBeta y
                  }

scaleGptGrads :: Double -> GptGrads -> GptGrads
scaleGptGrads s g = GptGrads
  { ggWte = scale s (ggWte g), ggWpe = scale s (ggWpe g)
  , ggBlocks = map (scaleBG s) (ggBlocks g)
  , ggLnGamma = scale s (ggLnGamma g), ggLnBeta = scale s (ggLnBeta g)
  , ggHead = scale s (ggHead g), ggHeadB = scale s (ggHeadB g)
  }
  where scaleBG s b = BlockGrads
          { bgAttnWq = scale s (bgAttnWq b), bgAttnWk = scale s (bgAttnWk b)
          , bgAttnWv = scale s (bgAttnWv b), bgAttnWo = scale s (bgAttnWo b)
          , bgAttnLnGamma = scale s (bgAttnLnGamma b), bgAttnLnBeta = scale s (bgAttnLnBeta b)
          , bgFfnW1 = scale s (bgFfnW1 b), bgFfnB1 = scale s (bgFfnB1 b)
          , bgFfnW2 = scale s (bgFfnW2 b), bgFfnB2 = scale s (bgFfnB2 b)
          , bgFfnLnGamma = scale s (bgFfnLnGamma b), bgFfnLnBeta = scale s (bgFfnLnBeta b)
          }

----------------------------------------------------------------------
-- Backward primitives (unchanged)
----------------------------------------------------------------------

linearBwd :: Matrix Double -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, Vector Double)
linearBwd x w gradY =
  let gradX = gradY LA.<> tr w
      gradW = tr x LA.<> gradY
      gradB = fromList [sumElements col | col <- LA.toColumns gradY]
  in  (gradX, gradW, gradB)

geluBwd :: Matrix Double -> Matrix Double -> Matrix Double
geluBwd x gradY =
  let xs = toList (LA.flatten x)
      gs = toList (LA.flatten gradY)
      deriv v = let a = 1.59577; b = 0.044715
                    z = a * v * (1 + b * v * v)
                    phi = 1 / (1 + exp (-z))
                    phi' = phi * (1 - phi) * a * (1 + 3 * b * v * v)
                in  phi + v * phi'
  in  reshape (cols x) (fromList (zipWith (*) gs (map deriv xs)))

softmaxBwd :: Matrix Double -> Matrix Double -> Matrix Double
softmaxBwd probs gradOut =
  fromRows $ zipWith softmaxRowBwd (toRows probs) (toRows gradOut)
  where softmaxRowBwd p go =
          let pv = toList p; gov = toList go; dot = sum (zipWith (*) pv gov)
          in  fromList $ zipWith (\pi goi -> pi * (goi - dot)) pv gov

layerNormBwd ::
  Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
  -> (Matrix Double, Vector Double, Vector Double)
layerNormBwd x gamma beta eps gradY =
  let d = fromIntegral (cols x)
      mu = LA.fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu
      var = LA.fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (\xi s -> scale s xi) (toRows xm) (toList invStd)
      gradXHat = fromRows $ zipWith (\go g -> scale g go) (toRows gradY) (toList gamma)
      gGamma = fromList [sumElements (go * xh) | (go, xh) <- zip (toRows gradY) (toRows xHat)]
      gBeta  = fromList [sumElements go | go <- toRows gradY]
      features = cols x
      gradX = fromRows [ gradRow (toRows gradXHat !! i) (toRows xm !! i)
                                   (toList invStd !! i) (toList var !! i)
                       | i <- [0 .. rows x - 1] ]
  in  (gradX, gGamma, gBeta)
  where
    gradRow gxh xi_minus_mu is v =
      let gInvStd = sumElements (gxh * xi_minus_mu)
          gVar = gInvStd * (-0.5) * (v + eps) ** (-1.5)
          gXm = scale is gxh + scale (2 * gVar / d) xi_minus_mu
          gMu = -sumElements gXm
      in  gXm + fromList (replicate features (gMu / d))
    d = fromIntegral (cols x); features = cols x

crossEntropyBwd :: Matrix Double -> [Int] -> (Double, Matrix Double)
crossEntropyBwd logits targetIds =
  let n = fromIntegral (rows logits)
      softmaxRows = map softmaxStable (toRows logits)
      losses = zipWith (\p t -> -log (max (toList p !! t) 1e-12)) softmaxRows targetIds
      totalLoss = sum losses / n
      gradRows = zipWith (\p t ->
        let pv = toList p
        in  fromList [if i == t then pv !! i - 1 else pv !! i | i <- [0 .. length pv - 1]]
        ) softmaxRows targetIds
  in  (totalLoss, scale (1 / n) (fromRows gradRows))
  where softmaxStable v = let mx = maxElement v; shifted = cmap (\x -> exp (x - mx)) v
                          in  scale (1 / sumElements shifted) shifted

----------------------------------------------------------------------
-- Multi-head attention backward (per-head, summed)
----------------------------------------------------------------------

-- | Backward through one head: x -> QKV -> scores -> softmax -> context.
--   gradOut is dL/d(context). Returns (grad_x, grad_Wq, grad_Wk, grad_Wv).
attentionHeadBwd :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
                 -> Matrix Double
                 -> (Matrix Double, Matrix Double, Matrix Double, Matrix Double)
attentionHeadBwd x wQ wK wV gradOut =
  let dk = fromIntegral (cols wQ) :: Double
      q = x LA.<> wQ; k = x LA.<> wK; v = x LA.<> wV
      scores = scale (1 / sqrt dk) (q LA.<> tr k)
      probs = softmaxStableF scores
      -- Backward through context = probs @ V
      gradProbs = gradOut LA.<> tr v
      gradV = tr probs LA.<> gradOut
      -- Backward through softmax
      gradScores = softmaxBwd probs gradProbs
      gradScoresS = scale (1 / sqrt dk) gradScores
      -- Backward through scores = Q @ K^T
      gradQ = gradScoresS LA.<> k
      gradK = tr gradScoresS LA.<> q
      -- Backward through projections
      (gradXq, gWQ, _) = linearBwd x wQ gradQ
      (gradXk, gWK, _) = linearBwd x wK gradK
      (gradXv, gWV, _) = linearBwd x wV gradV
      gradX = gradXq + gradXk + gradXv
  in  (gradX, gWQ, gWK, gWV)

softmaxStableF :: Matrix Double -> Matrix Double
softmaxStableF m = fromRows
  [let mx = maxElement row; s = cmap (\x -> exp (x - mx)) row
   in  scale (1 / sumElements s) s | row <- toRows m]

----------------------------------------------------------------------
-- Block backward
----------------------------------------------------------------------

-- | Backward through one transformer block.
--   Given gradient w.r.t. block output, returns (grad_wrt_input, BlockGrads).
blockBackward ::
  Int -> Int -> Double -> TransformerBlock -> Matrix Double -> Matrix Double
  -> (Matrix Double, BlockGrads)
blockBackward nHead seqLen eps tb xIn gradOut =
  let hDim = cols (tbAttnWq tb) `div` nHead

      -- ---- Forward pass (recompute intermediates from xIn) ----
      -- LN1
      an = layerNormF xIn (tbAttnLnGamma tb) (tbAttnLnBeta tb) eps
      -- Per-head attention
      headFwd h =
        let wQh = subMatrixW (tbAttnWq tb) 0 (h * hDim) (rows (tbAttnWq tb)) hDim
            wKh = subMatrixW (tbAttnWk tb) 0 (h * hDim) (rows (tbAttnWk tb)) hDim
            wVh = subMatrixW (tbAttnWv tb) 0 (h * hDim) (rows (tbAttnWv tb)) hDim
            qh = an LA.<> wQh; kh = an LA.<> wKh; vh = an LA.<> wVh
            sc = scale (1 / sqrt (fromIntegral hDim :: Double)) (qh LA.<> tr kh)
            scm = sc + causalMaskF seqLen
            pr = softmaxStableF scm
        in  pr LA.<> vh
      headOuts = map headFwd [0 .. nHead - 1]
      ctx = concatColsF headOuts
      attnOut = ctx LA.<> tbAttnWo tb
      postAttn = xIn + attnOut
      -- LN2
      fn = layerNormF postAttn (tbFfnLnGamma tb) (tbFfnLnBeta tb) eps
      -- FFN
      ff = tbFfn tb
      preGelu = fn LA.<> ffW1 ff + broadcastBias (ffB1 ff) seqLen
      hGelu = geluF preGelu
      ffnOut = hGelu LA.<> ffW2 ff + broadcastBias (ffB2 ff) seqLen
      xOut = postAttn + ffnOut

      -- ---- Backward pass ----
      -- gradOut = dL/d(xOut)
      -- Residual 2 splits gradient
      gradPostAttn2 = gradOut  -- flows to postAttn
      gradFfnOut = gradOut      -- flows to ffnOut

      -- FFN backward
      (gradH, gW2, gB2) = linearBwd hGelu (ffW2 ff) gradFfnOut
      gradPreGelu = geluBwd preGelu gradH
      (gradFn, gW1, gB1) = linearBwd fn (ffW1 ff) gradPreGelu
      (gradPostAttn1, gFfnLnGamma, gFfnLnBeta) = layerNormBwd postAttn (tbFfnLnGamma tb) (tbFfnLnBeta tb) eps gradFn

      -- Combine FFN + residual gradients into postAttn
      gradPostAttn = gradPostAttn2 + gradPostAttn1

      -- Residual 1 splits: to xIn and to attnOut
      gradXfromRes1 = gradPostAttn
      gradAttnOut = gradPostAttn

      -- Attention output projection
      (gradCtx, gWo, _) = linearBwd ctx (tbAttnWo tb) gradAttnOut

      -- Per-head backward
      headBwd h =
        let goh = subMatrixW gradCtx 0 (h * hDim) seqLen hDim
            wQh = subMatrixW (tbAttnWq tb) 0 (h * hDim) nEmb hDim
            wKh = subMatrixW (tbAttnWk tb) 0 (h * hDim) nEmb hDim
            wVh = subMatrixW (tbAttnWv tb) 0 (h * hDim) nEmb hDim
        in  attentionHeadBwd an wQh wKh wVh goh
      headGrads = map headBwd [0 .. nHead - 1]
      gradAn = foldl1' (+) [gx | (gx, _, _, _) <- headGrads]
      gWqFull = accumulateCols (tbAttnWq tb) [(h * hDim, gwq) | (h, (_, gwq, _, _)) <- zip [0..] headGrads]
      gWkFull = accumulateCols (tbAttnWk tb) [(h * hDim, gwk) | (h, (_, _, gwk, _)) <- zip [0..] headGrads]
      gWvFull = accumulateCols (tbAttnWv tb) [(h * hDim, gwv) | (h, (_, _, _, gwv)) <- zip [0..] headGrads]

      -- LN1 backward
      (gradXfromAttn, gAttnLnGamma, gAttnLnBeta) = layerNormBwd xIn (tbAttnLnGamma tb) (tbAttnLnBeta tb) eps gradAn

      -- Total grad w.r.t. xIn
      gradX = gradXfromRes1 + gradXfromAttn

      blockGrad = BlockGrads
        { bgAttnWq = gWqFull, bgAttnWk = gWkFull, bgAttnWv = gWvFull, bgAttnWo = gWo
        , bgAttnLnGamma = gAttnLnGamma, bgAttnLnBeta = gAttnLnBeta
        , bgFfnW1 = gW1, bgFfnB1 = gB1, bgFfnW2 = gW2, bgFfnB2 = gB2
        , bgFfnLnGamma = gFfnLnGamma, bgFfnLnBeta = gFfnLnBeta
        }
  in  (gradX, blockGrad)
  where
    nEmb = cols (tbAttnWq tb)
    causalMaskF n = fromLists [[if j <= i then 0 else -1/0 | j <- [0..n-1]] | i <- [0..n-1]]

----------------------------------------------------------------------
-- Full GPT backward
----------------------------------------------------------------------

gptBackward :: GptConfig -> Gpt -> [Int] -> [Int] -> (Double, GptGrads)
gptBackward cfg model inputIds targetIds =
  let seqLen = length inputIds
      nEmb = gptNEmbd cfg; nHead = gptNHead cfg; nLayer = gptNLayer cfg
      vocab = gptVocabSize cfg; eps = 1e-5
      blocks = gptBlocks model

      -- Embeddings
      wte = gptWte model; wpe = gptWpe model
      tokEmb = fromRows [toRows wte !! i | i <- inputIds]
      posEmb = subMatrixW wpe 0 0 seqLen nEmb
      x0 = tokEmb + posEmb

      -- Forward through blocks, saving x at each stage
      goFwd :: Matrix Double -> [TransformerBlock] -> [Matrix Double] -> [Matrix Double]
      goFwd x [] acc = reverse (x : acc)
      goFwd x (tb:tbs) acc =
        let hDim = nEmb `div` nHead
            an = layerNormF x (tbAttnLnGamma tb) (tbAttnLnBeta tb) eps
            heads = [ let wQh = subMatrixW (tbAttnWq tb) 0 (h*hDim) nEmb hDim
                          wKh = subMatrixW (tbAttnWk tb) 0 (h*hDim) nEmb hDim
                          wVh = subMatrixW (tbAttnWv tb) 0 (h*hDim) nEmb hDim
                          qh = an LA.<> wQh; kh = an LA.<> wKh; vh = an LA.<> wVh
                          sc = scale (1 / sqrt (fromIntegral hDim)) (qh LA.<> tr kh)
                          pr = softmaxStableF (sc + causalMaskF seqLen)
                      in  pr LA.<> vh
                    | h <- [0 .. nHead - 1] ]
            ctx = concatColsF heads
            attnOut = ctx LA.<> tbAttnWo tb
            postAttn = x + attnOut
            fn = layerNormF postAttn (tbFfnLnGamma tb) (tbFfnLnBeta tb) eps
            ff = tbFfn tb
            preGelu = fn LA.<> ffW1 ff + broadcastBias (ffB1 ff) seqLen
            hGelu = geluF preGelu
            ffnOut = hGelu LA.<> ffW2 ff + broadcastBias (ffB2 ff) seqLen
            x' = postAttn + ffnOut
        in  goFwd x' tbs (x : acc)
      fwdStack = goFwd x0 blocks []  -- [x0, x1, ..., xn]

      -- Last output
      xN = head fwdStack  -- post-block output
      xPre = fwdStack !! 1  -- input to last block

      -- Final norm + output projection
      xFinal = case blocks of
        [] -> x0
        _  -> let lastTb = last blocks
                  hDim = nEmb `div` nHead
                  an = layerNormF xPre (tbAttnLnGamma lastTb) (tbAttnLnBeta lastTb) eps
                  heads = [ let wQh = subMatrixW (tbAttnWq lastTb) 0 (h*hDim) nEmb hDim
                                wKh = subMatrixW (tbAttnWk lastTb) 0 (h*hDim) nEmb hDim
                                wVh = subMatrixW (tbAttnWv lastTb) 0 (h*hDim) nEmb hDim
                                qh = an LA.<> wQh; kh = an LA.<> wKh; vh = an LA.<> wVh
                                sc = scale (1 / sqrt (fromIntegral hDim)) (qh LA.<> tr kh)
                                pr = softmaxStableF (sc + causalMaskF seqLen)
                            in  pr LA.<> vh
                          | h <- [0 .. nHead - 1] ]
                  ctx = concatColsF heads
                  attnOut = ctx LA.<> tbAttnWo lastTb
                  postAttn = xPre + attnOut
                  fn = layerNormF postAttn (tbFfnLnGamma lastTb) (tbFfnLnBeta lastTb) eps
                  ff = tbFfn lastTb
                  preGelu2 = fn LA.<> ffW1 ff + broadcastBias (ffB1 ff) seqLen
                  hGelu2 = geluF preGelu2
                  ffnOut2 = hGelu2 LA.<> ffW2 ff + broadcastBias (ffB2 ff) seqLen
              in  postAttn + ffnOut2

      xFinalNorm = layerNormF xFinal (gptLnGamma model) (gptLnBeta model) eps
      logits = xFinalNorm LA.<> gptHead model + broadcastBias (gptHeadB model) seqLen

      -- Loss
      (loss, gradLogits) = crossEntropyBwd logits targetIds

      -- Backward: output projection
      (gradXFinalNorm, gHead, gHeadB) = linearBwd xFinalNorm (gptHead model) gradLogits
      (gradXFinal', gLnfGamma, gLnfBeta) = layerNormBwd xFinal (gptLnGamma model) (gptLnBeta model) eps gradXFinalNorm

      -- Backward through blocks (in reverse)
      -- We have fwdStack = [xN, x_{n-1}, ..., x0] where xN is post-block, x_{n-1} is input to last block
      -- gradXFinal' is gradient w.r.t. xN
      goBwd :: Matrix Double -> [TransformerBlock] -> [Matrix Double] -> (Matrix Double, [BlockGrads])
      goBwd gradX [] _ = (gradX, [])
      goBwd gradX (tb:tbs) (xPrev:xs) =
        let (gradInput, blockGrad) = blockBackward nHead seqLen eps tb xPrev gradX
            (gradFinal, restGrads) = goBwd gradInput tbs xs
        in  (gradFinal, blockGrad : restGrads)
      goBwd gradX _ _ = (gradX, [])  -- shouldn't happen

      (gradEmbed, blockGradsRev) = goBwd gradXFinal' (reverse blocks) (init fwdStack)
      blockGrads = reverse blockGradsRev

      -- Embedding gradients
      gWte = embedBackward wte inputIds gradEmbed
      gWpeFull = zeroMatrix 1024 nEmb
      gWpe' = updateSubMatrix gWpeFull 0 0 gradEmbed

      grads = (zeroGptGrads cfg)
        { ggWte = gWte, ggWpe = gWpe'
        , ggBlocks = blockGrads
        , ggLnGamma = gLnfGamma, ggLnBeta = gLnfBeta
        , ggHead = gHead, ggHeadB = gHeadB
        }
  in  (loss, grads)
  where
    causalMaskF n = fromLists [[if j <= i then 0 else -1/0 | j <- [0..n-1]] | i <- [0..n-1]]

----------------------------------------------------------------------
-- Forward helpers
----------------------------------------------------------------------

layerNormF :: Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
layerNormF x gamma beta eps =
  let d = fromIntegral (cols x)
      mu = LA.fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu
      var = LA.fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (\xi s -> scale s xi) (toRows xm) (toList invStd)
  in  fromRows (zipWith (\xh g -> scale g xh) (toRows xHat) (toList gamma))
    + broadcastBias beta (rows x)

geluF :: Matrix Double -> Matrix Double
geluF x = cmap f x
  where f v = let z = 1.59577 * v * (1 + 0.044715 * v * v) in v / (1 + exp (-z))

concatColsF :: [Matrix Double] -> Matrix Double
concatColsF [a] = a
concatColsF (a:as) = a ||| concatColsF as
concatColsF [] = error "no heads"

broadcastBias :: Vector Double -> Int -> Matrix Double
broadcastBias b n = reshape (LA.size b) (fromList (concat (replicate n (toList b))))

zeroMatrix :: Int -> Int -> Matrix Double
zeroMatrix r c = reshape c (fromList (replicate (r * c) 0))

----------------------------------------------------------------------
-- Utilities
----------------------------------------------------------------------

subMatrixW :: Matrix Double -> Int -> Int -> Int -> Int -> Matrix Double
subMatrixW m r c rows' cols' = LA.subMatrix (r, c) (rows', cols') m

accumulateCols :: Matrix Double -> [(Int, Matrix Double)] -> Matrix Double
accumulateCols template patches =
  let full = zeroMatrix (rows template) (cols template)
  in  foldl' (\acc (offset, patch) ->
        reshape (cols acc) (fromList
          [ if c >= offset && c < offset + cols patch
            then (toList (LA.flatten acc) !! (r * cols acc + c)) +
                 (toList (LA.flatten patch) !! (r * cols patch + (c - offset)))
            else toList (LA.flatten acc) !! (r * cols acc + c)
          | r <- [0 .. rows acc - 1], c <- [0 .. cols acc - 1]
          ])
        ) full patches

embedBackward :: Matrix Double -> [Int] -> Matrix Double -> Matrix Double
embedBackward wte inputIds gradX =
  let rowsGradX = toRows gradX
      updateFn acc (i, tid) =
        let oldRow = toRows acc !! tid
            newRow = oldRow + (rowsGradX !! i)
            vals = [ if ri == tid then toList newRow !! ci
                     else toList (LA.flatten acc) !! (ri * cols acc + ci)
                   | ri <- [0 .. rows acc - 1], ci <- [0 .. cols acc - 1] ]
        in  reshape (cols acc) (fromList vals)
  in  foldl' updateFn (zeroMatrix (rows wte) (cols wte)) (zip [0..] inputIds)

updateSubMatrix :: Matrix Double -> Int -> Int -> Matrix Double -> Matrix Double
updateSubMatrix target r0 c0 patch =
  reshape (cols target) (fromList
    [ if ri >= r0 && ri < r0 + rows patch && ci >= c0 && ci < c0 + cols patch
      then toList (LA.flatten patch) !! ((ri - r0) * cols patch + (ci - c0))
      else toList (LA.flatten target) !! (ri * cols target + ci)
    | ri <- [0 .. rows target - 1], ci <- [0 .. cols target - 1] ])
