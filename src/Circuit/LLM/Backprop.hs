{-# LANGUAGE OverloadedStrings #-}

-- | Full backward pass through one transformer block.
--   This is the core of the training backprop chain.
module Circuit.LLM.Backprop
  ( -- * Full backward pass
    gptBackward,
    bertBackward,

    -- * Gradient types
    GptGrads (..),
    BlockGrads (..),
    zeroGptGrads,
    addGptGrads,
    scaleGptGrads,

    -- * Primitives (exported for testing)
    linearBwd,
    layerNormBwd,
    geluBwd,
    softmaxBwd,
    crossEntropyBwd,
    maskedCrossEntropyBwd,
  )
where

import Circuit.LLM.Diff
  ( BlockParams (..),
    DiffP,
    GptParams (..),
    bertDiffP,
    gptDiffP,
    gptParamsFromModel,
    primBackward,
    primForward,
    subMatrixW,
    zeroMatrix,
  )
import Circuit.LLM.GPT (Gpt (..), GptConfig (..))
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromList,
    fromRows,
    maxElement,
    reshape,
    rows,
    scale,
    sumElements,
    toList,
    toRows,
    tr,
  )
import Numeric.LinearAlgebra qualified as LA

----------------------------------------------------------------------
-- Gradient accumulators (unchanged)
----------------------------------------------------------------------

data GptGrads = GptGrads
  { ggWte :: !(Matrix Double),
    ggWpe :: !(Matrix Double),
    ggBlocks :: ![BlockGrads],
    ggLnGamma :: !(Vector Double),
    ggLnBeta :: !(Vector Double),
    ggHead :: !(Matrix Double),
    ggHeadB :: !(Vector Double)
  }

data BlockGrads = BlockGrads
  { bgAttnWq, bgAttnWk, bgAttnWv, bgAttnWo :: !(Matrix Double),
    bgAttnLnGamma, bgAttnLnBeta :: !(Vector Double),
    bgFfnW1, bgFfnW2 :: !(Matrix Double),
    bgFfnB1, bgFfnB2 :: !(Vector Double),
    bgFfnLnGamma, bgFfnLnBeta :: !(Vector Double)
  }

zeroGptGrads :: GptConfig -> GptGrads
zeroGptGrads cfg =
  let nEmb = gptNEmbd cfg
      nLayer = gptNLayer cfg
      vocab = gptVocabSize cfg
      ffMul = 4
      zM r c = reshape c (fromList (replicate (r * c) 0))
      zV n = fromList (replicate n 0)
   in GptGrads
        { ggWte = zM vocab nEmb,
          ggWpe = zM 1024 nEmb,
          ggBlocks =
            replicate nLayer $
              BlockGrads
                { bgAttnWq = zM nEmb nEmb,
                  bgAttnWk = zM nEmb nEmb,
                  bgAttnWv = zM nEmb nEmb,
                  bgAttnWo = zM nEmb nEmb,
                  bgAttnLnGamma = zV nEmb,
                  bgAttnLnBeta = zV nEmb,
                  bgFfnW1 = zM nEmb (ffMul * nEmb),
                  bgFfnB1 = zV (ffMul * nEmb),
                  bgFfnW2 = zM (ffMul * nEmb) nEmb,
                  bgFfnB2 = zV nEmb,
                  bgFfnLnGamma = zV nEmb,
                  bgFfnLnBeta = zV nEmb
                },
          ggLnGamma = zV nEmb,
          ggLnBeta = zV nEmb,
          ggHead = zM nEmb vocab,
          ggHeadB = zV vocab
        }

addGptGrads :: GptGrads -> GptGrads -> GptGrads
addGptGrads a b =
  let r1 = rows (ggWte a)
      c1 = cols (ggWte a)
      r2 = rows (ggWte b)
      c2 = cols (ggWte b)
   in if r1 /= r2 || c1 /= c2
        then error $ "addGptGrads Wte mismatch: (" ++ show r1 ++ "," ++ show c1 ++ ") vs (" ++ show r2 ++ "," ++ show c2 ++ ")"
        else
          GptGrads
            { ggWte = ggWte a + ggWte b,
              ggWpe = ggWpe a + ggWpe b,
              ggBlocks = zipWith addBG (ggBlocks a) (ggBlocks b),
              ggLnGamma = ggLnGamma a + ggLnGamma b,
              ggLnBeta = ggLnBeta a + ggLnBeta b,
              ggHead = ggHead a + ggHead b,
              ggHeadB = ggHeadB a + ggHeadB b
            }
  where
    addBG x y =
      let r1 = rows (bgAttnWq x)
          c1 = cols (bgAttnWq x)
          r2 = rows (bgAttnWq y)
          c2 = cols (bgAttnWq y)
       in if r1 /= r2 || c1 /= c2
            then error $ "addBG Wq mismatch: (" ++ show r1 ++ "," ++ show c1 ++ ") vs (" ++ show r2 ++ "," ++ show c2 ++ ")"
            else
              BlockGrads
                { bgAttnWq = bgAttnWq x + bgAttnWq y,
                  bgAttnWk = bgAttnWk x + bgAttnWk y,
                  bgAttnWv = bgAttnWv x + bgAttnWv y,
                  bgAttnWo = bgAttnWo x + bgAttnWo y,
                  bgAttnLnGamma = bgAttnLnGamma x + bgAttnLnGamma y,
                  bgAttnLnBeta = bgAttnLnBeta x + bgAttnLnBeta y,
                  bgFfnW1 = bgFfnW1 x + bgFfnW1 y,
                  bgFfnB1 = bgFfnB1 x + bgFfnB1 y,
                  bgFfnW2 = bgFfnW2 x + bgFfnW2 y,
                  bgFfnB2 = bgFfnB2 x + bgFfnB2 y,
                  bgFfnLnGamma = bgFfnLnGamma x + bgFfnLnGamma y,
                  bgFfnLnBeta = bgFfnLnBeta x + bgFfnLnBeta y
                }

scaleGptGrads :: Double -> GptGrads -> GptGrads
scaleGptGrads s g =
  GptGrads
    { ggWte = scale s (ggWte g),
      ggWpe = scale s (ggWpe g),
      ggBlocks = map (scaleBG s) (ggBlocks g),
      ggLnGamma = scale s (ggLnGamma g),
      ggLnBeta = scale s (ggLnBeta g),
      ggHead = scale s (ggHead g),
      ggHeadB = scale s (ggHeadB g)
    }
  where
    scaleBG s_ b =
      BlockGrads
        { bgAttnWq = scale s_ (bgAttnWq b),
          bgAttnWk = scale s_ (bgAttnWk b),
          bgAttnWv = scale s_ (bgAttnWv b),
          bgAttnWo = scale s_ (bgAttnWo b),
          bgAttnLnGamma = scale s_ (bgAttnLnGamma b),
          bgAttnLnBeta = scale s_ (bgAttnLnBeta b),
          bgFfnW1 = scale s_ (bgFfnW1 b),
          bgFfnB1 = scale s_ (bgFfnB1 b),
          bgFfnW2 = scale s_ (bgFfnW2 b),
          bgFfnB2 = scale s_ (bgFfnB2 b),
          bgFfnLnGamma = scale s_ (bgFfnLnGamma b),
          bgFfnLnBeta = scale s_ (bgFfnLnBeta b)
        }

----------------------------------------------------------------------
-- Backward primitives (unchanged)
----------------------------------------------------------------------

linearBwd :: Matrix Double -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, Vector Double)
linearBwd x w gradY =
  let gradX = gradY LA.<> tr w
      gradW = tr x LA.<> gradY
      gradB = fromList [sumElements col | col <- LA.toColumns gradY]
   in (gradX, gradW, gradB)

geluBwd :: Matrix Double -> Matrix Double -> Matrix Double
geluBwd x gradY =
  let xs = toList (LA.flatten x)
      gs = toList (LA.flatten gradY)
      deriv v =
        let a = 1.59577
            b = 0.044715
            z = a * v * (1 + b * v * v)
            phi = 1 / (1 + exp (-z))
            phi' = phi * (1 - phi) * a * (1 + 3 * b * v * v)
         in phi + v * phi'
   in reshape (cols x) (fromList (zipWith (*) gs (map deriv xs)))

softmaxBwd :: Matrix Double -> Matrix Double -> Matrix Double
softmaxBwd probs gradOut =
  fromRows $ zipWith softmaxRowBwd (toRows probs) (toRows gradOut)
  where
    softmaxRowBwd p go =
      let pv = toList p; gov = toList go; dot = sum (zipWith (*) pv gov)
       in fromList $ zipWith (\pi_ goi -> pi_ * (goi - dot)) pv gov

layerNormBwd ::
  Matrix Double ->
  Vector Double ->
  Vector Double ->
  Double ->
  Matrix Double ->
  (Matrix Double, Vector Double, Vector Double)
layerNormBwd x gamma _beta eps gradY =
  let mu = LA.fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu
      var = LA.fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (flip scale) (toRows xm) (toList invStd)
      gammaList = toList gamma
      gradXHat = fromRows [LA.fromList $ zipWith (*) (toList go) gammaList | go <- toRows gradY]
      gGamma = fromList [sumElements (go * xh) | (go, xh) <- zip (LA.toColumns gradY) (LA.toColumns xHat)]
      gBeta = fromList [sumElements go | go <- LA.toColumns gradY]
      gradX =
        fromRows
          [ gradRow
              (toRows gradXHat !! i)
              (toRows xm !! i)
              (toList invStd !! i)
              (toList var !! i)
          | i <- [0 .. rows x - 1]
          ]
   in (gradX, gGamma, gBeta)
  where
    d = fromIntegral (cols x)
    features = cols x
    gradRow gxh xi_minus_mu is v =
      let gInvStd = sumElements (gxh * xi_minus_mu)
          gVar = gInvStd * (-0.5) * (v + eps) ** (-1.5)
          gXm = scale is gxh + scale (2 * gVar / d) xi_minus_mu
          gMu = -sumElements gXm
       in gXm + fromList (replicate features (gMu / d))

crossEntropyBwd :: Matrix Double -> [Int] -> (Double, Matrix Double)
crossEntropyBwd logits targetIds =
  let n = fromIntegral (rows logits)
      softmaxRows = map softmaxStable (toRows logits)
      losses = zipWith (\p t -> -log (max (toList p !! t) 1e-12)) softmaxRows targetIds
      totalLoss = sum losses / n
      gradRows =
        zipWith
          ( \p t ->
              let pv = toList p
               in fromList [if i == t then pv !! i - 1 else pv !! i | i <- [0 .. length pv - 1]]
          )
          softmaxRows
          targetIds
   in (totalLoss, scale (1 / n) (fromRows gradRows))
  where
    softmaxStable v =
      let mx = maxElement v; shifted = cmap (\x -> exp (x - mx)) v
       in scale (1 / sumElements shifted) shifted

-- | Masked cross-entropy for BERT-style training.
--   The mask indicates which positions are supervised; loss and gradients are
--   averaged only over those positions.
maskedCrossEntropyBwd :: Matrix Double -> [Int] -> [Bool] -> (Double, Matrix Double)
maskedCrossEntropyBwd logits targetIds mask =
  let maskedCount = fromIntegral (length (filter id mask))
      vocab = cols logits
      softmaxRows = map softmaxStable (toRows logits)
      maskedLosses =
        zipWith3
          ( \p t m ->
              if m then -log (max (toList p !! t) 1e-12) else 0
          )
          softmaxRows
          targetIds
          mask
      totalLoss = if maskedCount == 0 then 0 else sum maskedLosses / maskedCount
      gradRows =
        zipWith
          ( \p (t, m) ->
              if m
                then
                  let pv = toList p
                   in fromList [if i == t then pv !! i - 1 else pv !! i | i <- [0 .. length pv - 1]]
                else fromList (replicate vocab 0)
          )
          softmaxRows
          (zip targetIds mask)
   in if maskedCount == 0
        then (0, zeroMatrix (rows logits) (cols logits))
        else (totalLoss, scale (1 / maskedCount) (fromRows gradRows))
  where
    softmaxStable v =
      let mx = maxElement v; shifted = cmap (\x -> exp (x - mx)) v
       in scale (1 / sumElements shifted) shifted

-- | Convert a flat 'BlockParams' (used as the parameter/gradient carrier in
--   'DiffP') back to the existing 'BlockGrads' record.
blockGradsFromParams :: BlockParams -> BlockGrads
blockGradsFromParams bp =
  BlockGrads
    { bgAttnWq = bpAttnWq bp,
      bgAttnWk = bpAttnWk bp,
      bgAttnWv = bpAttnWv bp,
      bgAttnWo = bpAttnWo bp,
      bgAttnLnGamma = bpAttnLnGamma bp,
      bgAttnLnBeta = bpAttnLnBeta bp,
      bgFfnW1 = bpFfnW1 bp,
      bgFfnB1 = bpFfnB1 bp,
      bgFfnW2 = bpFfnW2 bp,
      bgFfnB2 = bpFfnB2 bp,
      bgFfnLnGamma = bpFfnLnGamma bp,
      bgFfnLnBeta = bpFfnLnBeta bp
    }

----------------------------------------------------------------------
-- Full model backward passes
----------------------------------------------------------------------

-- | Shared embedding setup used by both GPT and BERT backward passes.
embedInput :: GptConfig -> Gpt -> [Int] -> (Matrix Double, Matrix Double, Matrix Double)
embedInput cfg model inputIds =
  let seqLen = length inputIds
      nEmb = gptNEmbd cfg
      wte = gptWte model
      wpe = gptWpe model
      tokEmb = fromRows [toRows wte !! i | i <- inputIds]
      posEmb = subMatrixW wpe 0 0 seqLen nEmb
      x0 = tokEmb + posEmb
   in (x0, wte, wpe)

gptBackward :: GptConfig -> Gpt -> [Int] -> [Int] -> (Double, GptGrads)
gptBackward cfg model inputIds targetIds =
  let seqLen = length inputIds
      nEmb = gptNEmbd cfg
      eps = 1e-5

      (x0, wte, _) = embedInput cfg model inputIds

      -- Run the differentiable GPT body
      gptP = gptDiffP cfg seqLen eps
      params = gptParamsFromModel model
      logits = primForward gptP params x0

      -- Loss and output gradient
      (loss, gradLogits) = crossEntropyBwd logits targetIds

      -- Backward through the whole GPT body
      (gradX0, gp) = primBackward gptP params x0 gradLogits

      -- Embedding gradients
      gWte = embedBackward wte inputIds gradX0
      gWpeFull = zeroMatrix 1024 nEmb
      gWpe' = updateSubMatrix gWpeFull 0 0 gradX0

      grads =
        (zeroGptGrads cfg)
          { ggWte = gWte,
            ggWpe = gWpe',
            ggBlocks = map blockGradsFromParams (gpBlocks gp),
            ggLnGamma = gpLnGamma gp,
            ggLnBeta = gpLnBeta gp,
            ggHead = gpHead gp,
            ggHeadB = gpHeadB gp
          }
   in (loss, grads)

-- | BERT-style masked-language-model backward pass.
--
--   * inputIds are the (possibly masked) token IDs fed to the model.
--   * targetIds are the original token IDs to reconstruct.
--   * mask indicates which positions are supervised.
--
-- The model uses bidirectional attention, so every position can attend to
-- every other position.
bertBackward :: GptConfig -> Gpt -> [Int] -> [Int] -> [Bool] -> (Double, GptGrads)
bertBackward cfg model inputIds targetIds mask =
  let seqLen = length inputIds
      nEmb = gptNEmbd cfg
      eps = 1e-5

      (x0, wte, _) = embedInput cfg model inputIds

      -- Run the differentiable bidirectional body
      bertP = bertDiffP cfg seqLen eps
      params = gptParamsFromModel model
      logits = primForward bertP params x0

      -- Masked loss and output gradient
      (loss, gradLogits) = maskedCrossEntropyBwd logits targetIds mask

      -- Backward through the whole BERT body
      (gradX0, gp) = primBackward bertP params x0 gradLogits

      -- Embedding gradients
      gWte = embedBackward wte inputIds gradX0
      gWpeFull = zeroMatrix 1024 nEmb
      gWpe' = updateSubMatrix gWpeFull 0 0 gradX0

      grads =
        (zeroGptGrads cfg)
          { ggWte = gWte,
            ggWpe = gWpe',
            ggBlocks = map blockGradsFromParams (gpBlocks gp),
            ggLnGamma = gpLnGamma gp,
            ggLnBeta = gpLnBeta gp,
            ggHead = gpHead gp,
            ggHeadB = gpHeadB gp
          }
   in (loss, grads)

----------------------------------------------------------------------
-- Utilities
----------------------------------------------------------------------

embedBackward :: Matrix Double -> [Int] -> Matrix Double -> Matrix Double
embedBackward wte inputIds gradX =
  let rowsGradX = toRows gradX
      updateFn acc (i, tid) =
        let oldRow = toRows acc !! tid
            newRow = oldRow + (rowsGradX !! i)
            vals =
              [ if ri == tid
                  then toList newRow !! ci
                  else toList (LA.flatten acc) !! (ri * cols acc + ci)
              | ri <- [0 .. rows acc - 1],
                ci <- [0 .. cols acc - 1]
              ]
         in reshape (cols acc) (fromList vals)
   in foldl' updateFn (zeroMatrix (rows wte) (cols wte)) (zip [0 ..] inputIds)

updateSubMatrix :: Matrix Double -> Int -> Int -> Matrix Double -> Matrix Double
updateSubMatrix target r0 c0 patch =
  reshape
    (cols target)
    ( fromList
        [ if ri >= r0 && ri < r0 + rows patch && ci >= c0 && ci < c0 + cols patch
            then toList (LA.flatten patch) !! ((ri - r0) * cols patch + (ci - c0))
            else toList (LA.flatten target) !! (ri * cols target + ci)
        | ri <- [0 .. rows target - 1],
          ci <- [0 .. cols target - 1]
        ]
    )
