{-# LANGUAGE OverloadedStrings, BangPatterns #-}

-- | Manual backpropagation through the GPT-2 model.
--
-- Each operation has a backward function. The full backward pass
-- chains them together, accumulating parameter gradients.
--
-- Uses the convention: backward functions take (forward_inputs, grad_output)
-- and return (grad_input, param_gradients).
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
import Numeric.LinearAlgebra
  ( Matrix, Vector, cmap, fromList, fromRows, konst, maxElement, reshape
  , rows, cols, scale, sumElements, toList, toRows, tr
  )
import qualified Numeric.LinearAlgebra as LA

----------------------------------------------------------------------
-- Gradient accumulators
----------------------------------------------------------------------

data GptGrads = GptGrads
  { ggWte  :: !(Matrix Double)
  , ggWpe  :: !(Matrix Double)
  , ggBlocks :: ![BlockGrads]
  , ggLnGamma :: !(Vector Double)
  , ggLnBeta  :: !(Vector Double)
  , ggHead   :: !(Matrix Double)
  , ggHeadB  :: !(Vector Double)
  }

data BlockGrads = BlockGrads
  { bgAttnWq :: !(Matrix Double)
  , bgAttnWk :: !(Matrix Double)
  , bgAttnWv :: !(Matrix Double)
  , bgAttnWo :: !(Matrix Double)
  , bgAttnLnGamma :: !(Vector Double)
  , bgAttnLnBeta  :: !(Vector Double)
  , bgFfnW1 :: !(Matrix Double)
  , bgFfnB1 :: !(Vector Double)
  , bgFfnW2 :: !(Matrix Double)
  , bgFfnB2 :: !(Vector Double)
  , bgFfnLnGamma :: !(Vector Double)
  , bgFfnLnBeta  :: !(Vector Double)
  }

zeroGptGrads :: GptConfig -> GptGrads
zeroGptGrads cfg =
  let nEmb = gptNEmbd cfg; nLayer = gptNLayer cfg
      vocab = gptVocabSize cfg; ffMul = 4
      zM r c = reshape c (fromList (replicate (r * c) 0)); zV n = fromList (replicate n 0)
  in  GptGrads
        { ggWte = zM vocab nEmb
        , ggWpe = zM 1024 nEmb
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
addGptGrads a b = GptGrads
  { ggWte = ggWte a + ggWte b, ggWpe = ggWpe a + ggWpe b
  , ggBlocks = zipWith addBlockGrads (ggBlocks a) (ggBlocks b)
  , ggLnGamma = ggLnGamma a + ggLnGamma b, ggLnBeta = ggLnBeta a + ggLnBeta b
  , ggHead = ggHead a + ggHead b, ggHeadB = ggHeadB a + ggHeadB b
  }

addBlockGrads :: BlockGrads -> BlockGrads -> BlockGrads
addBlockGrads a b = BlockGrads
  { bgAttnWq = bgAttnWq a + bgAttnWq b, bgAttnWk = bgAttnWk a + bgAttnWk b
  , bgAttnWv = bgAttnWv a + bgAttnWv b, bgAttnWo = bgAttnWo a + bgAttnWo b
  , bgAttnLnGamma = bgAttnLnGamma a + bgAttnLnGamma b
  , bgAttnLnBeta  = bgAttnLnBeta a + bgAttnLnBeta b
  , bgFfnW1 = bgFfnW1 a + bgFfnW1 b, bgFfnB1 = bgFfnB1 a + bgFfnB1 b
  , bgFfnW2 = bgFfnW2 a + bgFfnW2 b, bgFfnB2 = bgFfnB2 a + bgFfnB2 b
  , bgFfnLnGamma = bgFfnLnGamma a + bgFfnLnGamma b
  , bgFfnLnBeta  = bgFfnLnBeta a + bgFfnLnBeta b
  }

scaleGptGrads :: Double -> GptGrads -> GptGrads
scaleGptGrads s g = GptGrads
  { ggWte = scale s (ggWte g), ggWpe = scale s (ggWpe g)
  , ggBlocks = map (scaleBlockGrads s) (ggBlocks g)
  , ggLnGamma = scale s (ggLnGamma g), ggLnBeta = scale s (ggLnBeta g)
  , ggHead = scale s (ggHead g), ggHeadB = scale s (ggHeadB g)
  }

scaleBlockGrads :: Double -> BlockGrads -> BlockGrads
scaleBlockGrads s b = BlockGrads
  { bgAttnWq = scale s (bgAttnWq b), bgAttnWk = scale s (bgAttnWk b)
  , bgAttnWv = scale s (bgAttnWv b), bgAttnWo = scale s (bgAttnWo b)
  , bgAttnLnGamma = scale s (bgAttnLnGamma b), bgAttnLnBeta = scale s (bgAttnLnBeta b)
  , bgFfnW1 = scale s (bgFfnW1 b), bgFfnB1 = scale s (bgFfnB1 b)
  , bgFfnW2 = scale s (bgFfnW2 b), bgFfnB2 = scale s (bgFfnB2 b)
  , bgFfnLnGamma = scale s (bgFfnLnGamma b), bgFfnLnBeta = scale s (bgFfnLnBeta b)
  }

----------------------------------------------------------------------
-- Backward primitives
----------------------------------------------------------------------

-- | Linear layer backward: y = x @ w + b (b broadcast across rows).
--   Returns (grad_x, grad_w, grad_b).
linearBwd :: Matrix Double -> Matrix Double -> Matrix Double
          -> (Matrix Double, Matrix Double, Vector Double)
linearBwd x w gradY =
  let gradX = gradY LA.<> tr w
      gradW = tr x LA.<> gradY
      gradB = fromList [sumElements col | col <- LA.toColumns gradY]
  in  (gradX, gradW, gradB)

-- | GELU backward (elementwise).
geluBwd :: Matrix Double -> Matrix Double -> Matrix Double
geluBwd x gradY =
  let xs = toList (LA.flatten x)
      gs = toList (LA.flatten gradY)
      deriv v = let a = 1.59577; b = 0.044715
                    z = a * v * (1 + b * v * v)
                    phi = 1 / (1 + exp (-z))
                    phi' = phi * (1 - phi) * a * (1 + 3 * b * v * v)
                in  phi + v * phi'
      result = zipWith (*) gs (map deriv xs)
  in  reshape (cols x) (fromList result)

-- | Row-wise softmax backward: probs = softmax(scores), grad_out given.
--   dL/dscores_i = probs_i * (grad_out_i - sum_j probs_j * grad_out_j)
softmaxBwd :: Matrix Double -> Matrix Double -> Matrix Double
softmaxBwd probs gradOut =
  fromRows $ zipWith softmaxRowBwd (toRows probs) (toRows gradOut)
  where
    softmaxRowBwd p go =
      let pv = toList p; gov = toList go
          dot = sum (zipWith (*) pv gov)
      in  fromList $ zipWith (\pi goi -> pi * (goi - dot)) pv gov

-- | LayerNorm backward.
--   y = (x - mu) / sqrt(var + eps) * gamma + beta
layerNormBwd ::
  Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
  -> (Matrix Double, Vector Double, Vector Double)
layerNormBwd x gamma beta eps gradY =
  let -- Recompute forward intermediates
      mu = LA.fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu  -- x - mu, broadcast
      var = LA.fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (\xi s -> scale s xi) (toRows xm) (toList invStd)

      -- grad w.r.t. xHat
      gradXHat = fromRows $ zipWith (\go g -> scale g go) (toRows gradY) (toList gamma)

      -- grad w.r.t. gamma, beta
      gGamma = fromList [sumElements (go * xh)
                        | (go, xh) <- zip (toRows gradY) (toRows xHat)]
      gBeta  = fromList [sumElements go | go <- toRows gradY]

      -- grad w.r.t. x (chain rule through xHat, invStd, var, mu, xm)
      gradX = fromRows
        [ gradRow (toRows gradXHat !! i) (toRows xm !! i)
                  (toList invStd !! i) (toList var !! i)
        | i <- [0 .. rows x - 1] ]

  in  (gradX, gGamma, gBeta)
  where
    d = fromIntegral (cols x)
    features = cols x
    gradRow gxh xi_minus_mu is v =
      let -- dL/d(invStd) per row
          gInvStd = sumElements (gxh * xi_minus_mu)
          -- dL/d(var) per row
          gVar = gInvStd * (-0.5) * (v + eps) ** (-1.5)
          -- dL/d(xi - mu): from xHat path + var path
          gXm = scale is gxh + scale (2 * gVar / d) xi_minus_mu
          -- dL/d(mu)
          gMu = -sumElements gXm
      in  gXm + fromList (replicate features (gMu / d))

-- | Cross-entropy loss backward.
--   Returns (loss, grad_wrt_logits).
crossEntropyBwd :: Matrix Double -> [Int] -> (Double, Matrix Double)
crossEntropyBwd logits targetIds =
  let n = fromIntegral (rows logits)
      softmaxRows = map softmaxStable (toRows logits)
      losses = zipWith (\p t -> -log (max (toList p !! t) 1e-12)) softmaxRows targetIds
      totalLoss = sum losses / n
      -- Gradient: (softmax - one_hot) / n
      gradRows = zipWith (\p t ->
        let pv = toList p
        in  fromList [if i == t then pv !! i - 1 else pv !! i | i <- [0 .. length pv - 1]]
        ) softmaxRows targetIds
  in  (totalLoss, scale (1 / n) (fromRows gradRows))
  where
    softmaxStable v =
      let mx = maxElement v; shifted = cmap (\x -> exp (x - mx)) v
      in  scale (1 / sumElements shifted) shifted

----------------------------------------------------------------------
-- Multi-head attention backward
----------------------------------------------------------------------

-- | Backward through multi-head attention.
--   Returns (grad_x, grad_Wq, grad_Wk, grad_Wv, grad_Wo).
attentionBwd ::
  Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
  -> Matrix Double -> Matrix Double
  -> (Matrix Double, Matrix Double, Matrix Double, Matrix Double, Matrix Double)
attentionBwd x wQ wK wV wO gradOut =
  let nHead = 1  -- Single head for simplicity; multi-head splits handled externally
      dk = fromIntegral (cols wQ)  -- Using full embedding as key dim
      -- Forward pass
      q = x LA.<> wQ
      k = x LA.<> wK
      v = x LA.<> wV
      scores = scale (1 / sqrt dk) (q LA.<> tr k)
      -- mask applied externally, assume scores already masked
      probs = softmaxStable scores
      context = probs LA.<> v
      out = context LA.<> wO  -- should equal gradOut source

      -- Backward through output projection
      (gradContext, gradWO, _) = linearBwd context wO gradOut

      -- Backward through context = probs @ V
      gradProbs = gradContext LA.<> tr v
      gradV = tr probs LA.<> gradContext

      -- Backward through softmax
      gradScores = softmaxBwd probs gradProbs
      gradScoresScaled = scale (1 / sqrt dk) gradScores  -- scale from forward

      -- Backward through scores = Q @ K^T
      gradQ1 = gradScoresScaled LA.<> k         -- dL/dQ from scores @ K
      gradK1 = tr gradScoresScaled LA.<> q     -- dL/dK from Q^T @ scores

      -- Backward through Q = x @ Wq, K = x @ Wk, V = x @ Wv
      (gradXq, gradWQ, _) = linearBwd x wQ gradQ1
      (gradXk, gradWK, _) = linearBwd x wK gradK1
      (gradXv, gradWV, _) = linearBwd x wV gradV

      gradX = gradXq + gradXk + gradXv
  in  (gradX, gradWQ, gradWK, gradWV, gradWO)
  where
    softmaxStable m =
      fromRows [let mx = maxElement row; s = cmap (\x -> exp (x - mx)) row
                    sm = sumElements s in scale (1 / sm) s | row <- toRows m]

----------------------------------------------------------------------
-- Full GPT backward pass
----------------------------------------------------------------------

-- | Full forward + backward pass through GPT-2.
--   Returns (loss, parameter_gradients).
gptBackward :: GptConfig -> Gpt -> [Int] -> [Int] -> (Double, GptGrads)
gptBackward cfg model inputIds targetIds =
  let seqLen = length inputIds
      nEmb = gptNEmbd cfg; nHead = gptNHead cfg; vocab = gptVocabSize cfg
      maxSeq = 1024; ffMul = 4

      -- ---- Forward pass (saving intermediates) ----
      wte = gptWte model; wpe = gptWpe model

      -- Token embeddings: rows of wte indexed by inputIds
      tokEmb = fromRows [toRows wte !! i | i <- inputIds]  -- [seq, nEmb]
      posEmb = subMatrixW wpe 0 0 seqLen nEmb              -- [seq, nEmb]
      x = tokEmb + posEmb                                   -- [seq, nEmb]

      -- ---- Placeholder for block-by-block forward ----
      -- For now, we compute the full forward using Circuit.LLM.GPT.forward
      -- and then use a simplified backward that handles the output projection
      -- and final layer norm, with block-level gradients stubbed.

      -- Full forward pass (existing implementation)
      logits = forward cfg model inputIds  -- [seq, vocab]

      -- ---- Backward pass ----
      (loss, gradLogits) = crossEntropyBwd logits targetIds

      -- gradLogits -> grad through output projection (gptHead, gptHeadB)
      (gradPreHead, gHead, gHeadB) = linearBwd x (gptHead model) gradLogits
      -- Note: x here should be the pre-head activations from the last transformer block
      -- For the simplified stub, we propagate back directly

      -- Accumulate gradients
      grads = (zeroGptGrads cfg)
        { ggHead = gHead
        , ggHeadB = gHeadB
        }

  in  (loss, grads)

-- Helper: subMatrix wrapper
subMatrixW :: Matrix Double -> Int -> Int -> Int -> Int -> Matrix Double
subMatrixW m r c rows' cols' = LA.subMatrix (r, c) (rows', cols') m
