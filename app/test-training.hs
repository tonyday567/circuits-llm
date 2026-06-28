{-# LANGUAGE OverloadedStrings #-}
module Main where

import Circuit.LLM.GPT (GptConfig (..), forward)
import Circuit.LLM.Training (crossEntropyLoss)
import Circuit.LLM.Weights (loadGpt2With)
import Control.Exception (catch, IOException)

main :: IO ()
main = do
  let cfg = GptConfig
        { gptVocabSize = 1000
        , gptNEmbd     = 32
        , gptNHead     = 4
        , gptNLayer    = 2
        }

  putStrLn "Loading weights..."
  m <- loadGpt2With "/tmp/gpt2-weights" cfg
       `catch` \(_ :: IOException) -> error "Need synthetic weights at /tmp/gpt2-weights"

  putStrLn "Testing cross-entropy loss..."
  let logits = forward cfg m [1, 2, 3, 4]
      targets = [2, 3, 4, 5]
      loss = crossEntropyLoss logits targets
  putStrLn $ "  loss = " ++ show loss
  putStrLn "OK"
