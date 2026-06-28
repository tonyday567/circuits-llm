{-# LANGUAGE OverloadedStrings #-}
module Main where

import Circuit.LLM.GPT (GptConfig (..), forward)
import Circuit.LLM.Inference (greedySample, lastRow)
import Circuit.LLM.Weights (loadGpt2With)
import Numeric.LinearAlgebra (sumElements)

main :: IO ()
main = do
  let cfg = GptConfig
        { gptVocabSize = 1000
        , gptNEmbd     = 32
        , gptNHead     = 4
        , gptNLayer    = 2
        }

  putStrLn "Loading synthetic weights..."
  m <- loadGpt2With "/tmp/gpt2-weights" cfg

  putStrLn "Testing greedy sampling..."
  let logits = forward cfg m [1, 2, 3, 4]
      lastLogits = lastRow logits
      nextToken = greedySample lastLogits
      s = sumElements lastLogits
  putStrLn $ "  sum(last logits) = " ++ show s
  putStrLn $ "  next token = " ++ show nextToken

  putStrLn "Testing auto-regressive loop (5 steps)..."
  let go [] _ = pure []
      go toks 0 = pure toks
      go toks n = do
        let l = forward cfg m toks
            next = greedySample (lastRow l)
        go (toks ++ [next]) (n - 1)
  result <- go [1, 2, 3, 4] 5
  putStrLn $ "  generated tokens: " ++ show result
  putStrLn "OK"
