{-# LANGUAGE OverloadedStrings #-}

module Main where

import Circuit.LLM.Backprop (gptBackward)
import Circuit.LLM.GPT (GptConfig (..))
import Circuit.LLM.Training (adamwStep, initAdamW, trainLoop)
import Circuit.LLM.Weights (loadGpt2With)

main :: IO ()
main = do
  let cfg = GptConfig {gptVocabSize = 1000, gptNEmbd = 32, gptNHead = 4, gptNLayer = 2}
      lr = 1e-3
      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8
      wd = 0.01
      seqLen = 4
      steps = 3
      data_ = concat (replicate 50 [1 .. 10 :: Int])

  putStrLn "Loading synthetic weights..."
  m <- loadGpt2With "/tmp/gpt2-weights" cfg

  putStrLn $ "Training for " ++ show steps ++ " steps..."
  (_, losses) <- trainLoop lr beta1 beta2 eps wd cfg m data_ seqLen steps

  putStrLn "\nResults:"
  case (losses, reverse losses) of
    ([], _) -> putStrLn "  No losses recorded"
    (l0 : _, lf : _) -> do
      putStrLn $ "  Initial loss: " ++ show l0
      putStrLn $ "  Final loss:   " ++ show lf
  putStrLn "  All steps completed ✓"
