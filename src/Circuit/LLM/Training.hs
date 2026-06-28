{-# LANGUAGE OverloadedStrings #-}

-- | Training: loss function and training loop for GPT-2.
--
-- Currently: forward pass + cross-entropy loss, no backprop.
-- Backprop and optimizer are stubbed for incremental addition.
module Circuit.LLM.Training
  ( -- * Loss
    crossEntropyLoss

    -- * Training
  , trainStep
  , trainLoop
  ) where

import Circuit.LLM.GPT (GptConfig (..), Gpt (..), forward)
import Numeric.LinearAlgebra
  ( Matrix, Vector, cmap, maxElement, rows, sumElements, toList, toRows )
import Text.Printf (printf)

----------------------------------------------------------------------
-- Cross-entropy loss
----------------------------------------------------------------------

-- | Cross-entropy loss for next-token prediction.
--   logits: [seq_len, vocab_size]   targets: [seq_len] token IDs
--   Returns the average loss.
crossEntropyLoss :: Matrix Double -> [Int] -> Double
crossEntropyLoss logits targetIds =
  let n = fromIntegral (rows logits)
      losses = zipWith crossEntropyRow (toRows logits) targetIds
  in  sum losses / n

-- | Cross-entropy for a single row: -log(softmax(row)[target]).
crossEntropyRow :: Vector Double -> Int -> Double
crossEntropyRow row target =
  let mx = maxElement row
      shifted = cmap (\x -> exp (x - mx)) row
      s = sumElements shifted
      pTarget = (toList shifted !! target) / s
  in  -log (max pTarget 1e-12)  -- clamp to avoid log(0)

----------------------------------------------------------------------
-- Training loop
----------------------------------------------------------------------

-- | One training step: forward pass, compute loss.
trainStep :: GptConfig -> Gpt -> [Int] -> [Int] -> IO Double
trainStep cfg model inputIds targetIds = do
  let logits = forward cfg model inputIds
      loss = crossEntropyLoss logits targetIds
  pure loss

-- | Simple training loop over tokenized text data.
--   data_ is a flat list of token IDs. We slide a window of seqLen tokens.
trainLoop :: GptConfig -> Gpt -> [Int] -> Int -> Int -> IO ()
trainLoop cfg model data_ seqLen steps = go model steps 0
  where
    go _ 0 _ = pure ()
    go m n offset = do
      let endIdx = offset + seqLen + 1  -- +1 for target shift
      if endIdx > length data_
        then go m (n - 1) 0  -- wrap around
        else do
          let inputs = take seqLen (drop offset data_)
              targets = take seqLen (drop (offset + 1) data_)
          loss <- trainStep cfg m inputs targets
          printf "  step %d: loss=%.6f\n" (steps - n + 1) loss
          go m (n - 1) (offset + seqLen)
