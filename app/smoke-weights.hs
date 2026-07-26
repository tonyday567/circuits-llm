{-# LANGUAGE OverloadedStrings #-}

module Main where

import Circuit.LLM.GPT (GptConfig (..), forward)
import Circuit.LLM.Weights (loadGpt2With)
import Numeric.LinearAlgebra (sumElements)
import Prelude hiding (sum)

main :: IO ()
main = do
  let cfg =
        GptConfig
          { gptVocabSize = 1000,
            gptNEmbd = 32,
            gptNHead = 4,
            gptNLayer = 2
          }

  putStrLn "Loading synthetic weights from /tmp/gpt2-weights..."
  m <- loadGpt2With "/tmp/gpt2-weights" cfg

  let inputIds = [1 .. 8]

  putStrLn "Running forward pass with loaded weights..."
  let logits = forward cfg m inputIds
      s = sumElements logits
  putStrLn $ "  sum(logits) = " ++ show s
  putStrLn "OK"
