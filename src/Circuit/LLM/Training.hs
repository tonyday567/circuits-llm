{-# LANGUAGE OverloadedStrings #-}

-- | Training: loss, backprop, AdamW optimizer, and training loop.
module Circuit.LLM.Training
  ( -- * Optimizer
    AdamWState,
    initAdamW,
    adamwStep,

    -- * Training
    trainStep,
    trainLoop,
    crossEntropyLoss,

    -- * Masked LM (BERT-style)
    trainStepMasked,
    trainLoopMasked,
    maskPositions,
    applyMask,
  )
where

import Circuit.LLM.Backprop
  ( BlockGrads (..),
    GptGrads (..),
    addGptGrads,
    bertBackward,
    crossEntropyBwd,
    gptBackward,
    scaleGptGrads,
    zeroGptGrads,
  )
import Circuit.LLM.GPT
  ( FeedForward (..),
    Gpt (..),
    GptConfig (..),
    TransformerBlock (..),
  )
import Control.Monad (when)
import Debug.Trace (trace)
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromList,
    reshape,
    rows,
    scale,
    sumElements,
    toList,
    toRows,
    tr,
  )
import Numeric.LinearAlgebra qualified as LA
import System.Random (newStdGen, randomRs)
import Text.Printf (printf)

----------------------------------------------------------------------
-- AdamW optimizer
----------------------------------------------------------------------

-- | AdamW optimizer state: step count, first moments, second moments.
data AdamWState = AdamWState
  { adamT :: !Int,
    adamM :: !GptGrads,
    adamV :: !GptGrads
  }

-- | Initialize AdamW state.
initAdamW :: GptConfig -> AdamWState
initAdamW cfg =
  AdamWState
    { adamT = 0,
      adamM = zeroGptGrads cfg,
      adamV = zeroGptGrads cfg
    }

-- | One AdamW step: update model parameters given gradients.
adamwStep ::
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  GptConfig ->
  AdamWState ->
  Gpt ->
  GptGrads ->
  (AdamWState, Gpt)
adamwStep lr beta1 beta2 eps wd cfg state model grads =
  let t = adamT state + 1
      m = trace "momentUpdate..." $ momentUpdate beta1 (adamM state) grads
      v = trace "momentUpdate2..." $ momentUpdate2 beta2 (adamV state) grads
      lr_t = lr * sqrt (1 - beta2 ^^ t) / (1 - beta1 ^^ t)
      model' = trace "applyUpdates..." $ applyUpdates lr_t eps (wd * lr) cfg model m v
      state' = AdamWState {adamT = t, adamM = m, adamV = v}
   in (state', model')

-- | Update first moment: m = beta1 * m + (1 - beta1) * grad.
momentUpdate :: Double -> GptGrads -> GptGrads -> GptGrads
momentUpdate beta m g =
  addGptGrads (scaleGptGrads beta m) (scaleGptGrads (1 - beta) g)

-- | Update second moment: v = beta2 * v + (1 - beta2) * grad^2.
momentUpdate2 :: Double -> GptGrads -> GptGrads -> GptGrads
momentUpdate2 beta v g =
  let sg = trace "squareGrads..." $ squareGrads g
      scaled = trace "scaleGrads..." $ scaleGptGrads (1 - beta) sg
   in trace "momentUpdate2 add..." $ addGptGrads (scaleGptGrads beta v) scaled

-- | Element-wise square of gradients (for second moment).
squareGrads :: GptGrads -> GptGrads
squareGrads g =
  let ok = rows (ggWte g) > 0
   in if not ok
        then error "squareGrads: empty Wte"
        else
          g
            { ggWte = ggWte g * ggWte g,
              ggWpe = ggWpe g * ggWpe g,
              ggBlocks = map squareBlockGrads (ggBlocks g),
              ggLnGamma = ggLnGamma g * ggLnGamma g,
              ggLnBeta = ggLnBeta g * ggLnBeta g,
              ggHead = ggHead g * ggHead g,
              ggHeadB = ggHeadB g * ggHeadB g
            }

squareBlockGrads :: BlockGrads -> BlockGrads
squareBlockGrads b =
  b
    { bgAttnWq = bgAttnWq b * bgAttnWq b,
      bgAttnWk = bgAttnWk b * bgAttnWk b,
      bgAttnWv = bgAttnWv b * bgAttnWv b,
      bgAttnWo = bgAttnWo b * bgAttnWo b,
      bgAttnLnGamma = bgAttnLnGamma b * bgAttnLnGamma b,
      bgAttnLnBeta = bgAttnLnBeta b * bgAttnLnBeta b,
      bgFfnW1 = bgFfnW1 b * bgFfnW1 b,
      bgFfnB1 = bgFfnB1 b * bgFfnB1 b,
      bgFfnW2 = bgFfnW2 b * bgFfnW2 b,
      bgFfnB2 = bgFfnB2 b * bgFfnB2 b,
      bgFfnLnGamma = bgFfnLnGamma b * bgFfnLnGamma b,
      bgFfnLnBeta = bgFfnLnBeta b * bgFfnLnBeta b
    }

-- | Apply AdamW updates to all model parameters.
applyUpdates ::
  Double -> Double -> Double -> GptConfig -> Gpt -> GptGrads -> GptGrads -> Gpt
applyUpdates lr_t eps wd_lr cfg model m v =
  let m_ggWte = ggWte m
      v_ggWte = ggWte v
      m_ggWpe = ggWpe m
      v_ggWpe = ggWpe v
   in model
        { gptWte = updateMatrix lr_t eps wd_lr (gptWte model) (ggWte m) (ggWte v),
          gptWpe = updateMatrix lr_t eps wd_lr (gptWpe model) (ggWpe m) (ggWpe v),
          gptBlocks =
            zipWith3
              (updateBlock lr_t eps wd_lr)
              (gptBlocks model)
              (ggBlocks m)
              (ggBlocks v),
          gptLnGamma = updateVector lr_t eps wd_lr (gptLnGamma model) (ggLnGamma m) (ggLnGamma v),
          gptLnBeta = updateVector lr_t eps wd_lr (gptLnBeta model) (ggLnBeta m) (ggLnBeta v),
          gptHead = updateMatrix lr_t eps wd_lr (gptHead model) (ggHead m) (ggHead v),
          gptHeadB = updateVector lr_t eps wd_lr (gptHeadB model) (ggHeadB m) (ggHeadB v)
        }

updateBlock :: Double -> Double -> Double -> TransformerBlock -> BlockGrads -> BlockGrads -> TransformerBlock
updateBlock lr_t eps wd_lr tb m v =
  tb
    { tbAttnWq = updateMatrix lr_t eps wd_lr (tbAttnWq tb) (bgAttnWq m) (bgAttnWq v),
      tbAttnWk = updateMatrix lr_t eps wd_lr (tbAttnWk tb) (bgAttnWk m) (bgAttnWk v),
      tbAttnWv = updateMatrix lr_t eps wd_lr (tbAttnWv tb) (bgAttnWv m) (bgAttnWv v),
      tbAttnWo = updateMatrix lr_t eps wd_lr (tbAttnWo tb) (bgAttnWo m) (bgAttnWo v),
      tbAttnLnGamma = updateVector lr_t eps wd_lr (tbAttnLnGamma tb) (bgAttnLnGamma m) (bgAttnLnGamma v),
      tbAttnLnBeta = updateVector lr_t eps wd_lr (tbAttnLnBeta tb) (bgAttnLnBeta m) (bgAttnLnBeta v),
      tbFfn =
        let ff = tbFfn tb; mf = m; vf = v
         in ff
              { ffW1 = updateMatrix lr_t eps wd_lr (ffW1 ff) (bgFfnW1 mf) (bgFfnW1 vf),
                ffB1 = updateVector lr_t eps wd_lr (ffB1 ff) (bgFfnB1 mf) (bgFfnB1 vf),
                ffW2 = updateMatrix lr_t eps wd_lr (ffW2 ff) (bgFfnW2 mf) (bgFfnW2 vf),
                ffB2 = updateVector lr_t eps wd_lr (ffB2 ff) (bgFfnB2 mf) (bgFfnB2 vf)
              },
      tbFfnLnGamma = updateVector lr_t eps wd_lr (tbFfnLnGamma tb) (bgFfnLnGamma m) (bgFfnLnGamma v),
      tbFfnLnBeta = updateVector lr_t eps wd_lr (tbFfnLnBeta tb) (bgFfnLnBeta m) (bgFfnLnBeta v)
    }

-- | AdamW update for one matrix: param -= lr_t * m / (sqrt(v) + eps) + wd_lr * param
updateMatrix :: Double -> Double -> Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
updateMatrix lr_t eps wd_lr param m_ v_ =
  let nParams = rows param * cols param
      nM = rows m_ * cols m_
      nV = rows v_ * cols v_
   in if nParams /= nM || nParams /= nV
        then
          error $
            "updateMatrix: param="
              ++ show (rows param, cols param)
              ++ " m="
              ++ show (rows m_, cols m_)
              ++ " v="
              ++ show (rows v_, cols v_)
        else
          let pv = toList (LA.flatten param)
              mv = toList (LA.flatten m_)
              vv = toList (LA.flatten v_)
           in reshape
                (cols param)
                ( fromList $
                    zipWith3 (\p m v -> p - lr_t * m / (sqrt v + eps) - wd_lr * p) pv mv vv
                )

-- | AdamW update for one vector.
updateVector :: Double -> Double -> Double -> Vector Double -> Vector Double -> Vector Double -> Vector Double
updateVector lr_t eps wd_lr param m_ v_ =
  let pv = toList param
      mv = toList m_
      vv = toList v_
      np = length pv
      nm = length mv
      nv = length vv
   in if np /= nm || np /= nv
        then error $ "updateVector size mismatch: param=" ++ show np ++ " m=" ++ show nm ++ " v=" ++ show nv
        else fromList $ zipWith3 (\p m v -> p - lr_t * m / (sqrt v + eps) - wd_lr * p) pv mv vv

-- | AdamW update for one matrix.
-- Training loop

----------------------------------------------------------------------

-- | Cross-entropy loss from logits and target token IDs.
crossEntropyLoss :: Matrix Double -> [Int] -> Double
crossEntropyLoss logits targetIds = fst (crossEntropyBwd logits targetIds)

-- | One training step: forward + backward + loss, return (loss, gradients).
trainStep :: GptConfig -> Gpt -> [Int] -> [Int] -> IO (Double, GptGrads)
trainStep cfg model inputIds targetIds = do
  let (loss, grads) = gptBackward cfg model inputIds targetIds
  pure (loss, grads)

-- | Full training loop with AdamW optimization.
--   data_ is a flat list of token IDs. We slide a window of seqLen tokens.
--   Returns the trained model.
trainLoop ::
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  GptConfig ->
  Gpt ->
  [Int] ->
  Int ->
  Int ->
  IO (Gpt, [Double])
trainLoop lr beta1 beta2 eps wd cfg model data_ seqLen steps =
  let opt0 = initAdamW cfg
   in go model opt0 [] steps 0
  where
    go m opt losses 0 _ = pure (m, reverse losses)
    go m opt losses n offset = do
      let endIdx = offset + seqLen + 1
      if endIdx > length data_
        then go m opt losses (n - 1) 0
        else do
          let inputs = take seqLen (drop offset data_)
              targets = take seqLen (drop (offset + 1) data_)
          (loss, grads) <- trainStep cfg m inputs targets
          let (opt', m') = adamwStep lr beta1 beta2 eps wd cfg opt m grads
          let stepNum = steps - n + 1
          when (stepNum `mod` 10 == 0 || stepNum == 1) $
            printf "  step %d: loss=%.6f\n" stepNum loss
          go m' opt' (loss : losses) (n - 1) (offset + seqLen)

----------------------------------------------------------------------
-- Masked language model training (BERT-style)
----------------------------------------------------------------------

-- | One masked-LM training step: forward + backward + masked loss.
trainStepMasked :: GptConfig -> Gpt -> [Int] -> [Int] -> [Bool] -> IO (Double, GptGrads)
trainStepMasked cfg model inputIds targetIds mask = do
  let (loss, grads) = bertBackward cfg model inputIds targetIds mask
  pure (loss, grads)

-- | Randomly choose positions to mask.  At least one position is always
--   masked so the loss is well-defined.
maskPositions :: Int -> Double -> IO [Bool]
maskPositions seqLen maskRate = do
  gen <- newStdGen
  let nMask = max 1 (floor (fromIntegral seqLen * maskRate))
      idxs = take nMask (randomRs (0, seqLen - 1) gen)
  pure [i `elem` idxs | i <- [0 .. seqLen - 1]]

-- | Replace masked positions with a fixed mask token ID.
applyMask :: [Int] -> [Bool] -> Int -> [Int]
applyMask ids mask maskId = zipWith (\i m -> if m then maskId else i) ids mask

-- | Masked-LM training loop with AdamW optimization.
--
--   * maskRate is the fraction of positions to mask per step.
--   * maskId is the token ID used to represent the [MASK] token.
--   * data_ is a flat list of token IDs.  We slide a window of seqLen tokens.
--   * The targets are the original (unmasked) token IDs.
trainLoopMasked ::
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  GptConfig ->
  Gpt ->
  [Int] ->
  Int ->
  Int ->
  Double ->
  Int ->
  IO (Gpt, [Double])
trainLoopMasked lr beta1 beta2 eps wd cfg model data_ seqLen steps maskRate maskId =
  let opt0 = initAdamW cfg
   in go model opt0 [] steps 0
  where
    go m opt losses 0 _ = pure (m, reverse losses)
    go m opt losses n offset = do
      if offset + seqLen > length data_
        then go m opt losses (n - 1) 0
        else do
          let inputs = take seqLen (drop offset data_)
          mask <- maskPositions seqLen maskRate
          let maskedInputs = applyMask inputs mask maskId
          (loss, grads) <- trainStepMasked cfg m maskedInputs inputs mask
          let (opt', m') = adamwStep lr beta1 beta2 eps wd cfg opt m grads
          let stepNum = steps - n + 1
          when (stepNum `mod` 10 == 0 || stepNum == 1) $
            printf "  step %d: loss=%.6f\n" stepNum loss
          go m' opt' (loss : losses) (n - 1) (offset + seqLen)
