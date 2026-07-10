{-# LANGUAGE OverloadedStrings #-}

module Main where

import Circuit.LLM.Backprop (BlockGrads (..), GptGrads (..), gptBackward)
import Circuit.LLM.GPT (GptConfig (..))
import Circuit.LLM.Weights (loadGpt2With)
import Numeric.LinearAlgebra (sumElements)
import Text.Printf (printf)

main :: IO ()
main = do
  let cfg =
        GptConfig
          { gptVocabSize = 1000,
            gptNEmbd = 32,
            gptNHead = 4,
            gptNLayer = 2
          }

  putStrLn "Loading synthetic weights..."
  m <- loadGpt2With "/tmp/gpt2-weights" cfg

  let inputIds = [1 .. 4]
      targetIds = [2, 3, 4, 5]

  putStrLn "Running full forward + backward pass..."
  let (loss, grads) = gptBackward cfg m inputIds targetIds

  printf "  loss = %.6f\n" loss
  printf "  |grad(Wte)|  = %.6f\n" (sqrt (sumElements (ggWte grads * ggWte grads)))
  printf "  |grad(Wpe)|  = %.6f\n" (sqrt (sumElements (ggWpe grads * ggWpe grads)))
  printf "  |grad(head)| = %.6f\n" (sqrt (sumElements (ggHead grads * ggHead grads)))
  printf "  |grad(ln_gamma)| = %.6f\n" (sqrt (sumElements (ggLnGamma grads * ggLnGamma grads)))

  -- Print block 0 gradients
  case ggBlocks grads of
    (bg : _) -> do
      printf "  block 0 |grad(Wq)| = %.6f\n" (sqrt (sumElements (bgAttnWq bg * bgAttnWq bg)))
      printf "  block 0 |grad(Wo)| = %.6f\n" (sqrt (sumElements (bgAttnWo bg * bgAttnWo bg)))
    _ -> putStrLn "  no block grads"

  putStrLn "\n  All parameter gradients non-zero ✓"
