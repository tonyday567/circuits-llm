{-# LANGUAGE OverloadedStrings, BangPatterns #-}
module Main where

import Circuit.LLM.Attention
import Control.Exception (evaluate)
import Control.Monad (replicateM_)
import Data.Foldable (sum)
import Harpie.Array (Array, array)
import System.CPUTime (getCPUTime)
import Text.Printf (printf)

randArray :: Int -> Int -> Double -> Array Double
randArray rows cols seed = array [rows, cols] (take (rows * cols) values)
 where
  values = iterate nextR (sin seed * 0.5 + 0.5)
  nextR x = sin (x * 127.1 + 311.7) * 0.5 + 0.5

main :: IO ()
main = do
  let seqLen = 64; nEmbd = 64; nHead = 4; headDim = nEmbd `div` nHead
  putStrLn "=== Haskell Attention Benchmarks ==="
  printf "  config: seq_len=%d, n_embd=%d, n_head=%d\n\n" seqLen nEmbd nHead

  let x = randArray seqLen nEmbd 5.0
      wq = randArray nEmbd nEmbd 1.0
      wk = randArray nEmbd nEmbd 2.0
      wv = randArray nEmbd nEmbd 3.0
      wo = randArray nEmbd nEmbd 4.0
      q = randArray seqLen headDim 6.0
      k = randArray seqLen headDim 7.0
      v = randArray seqLen headDim 8.0
      mask = causalMask seqLen

  -- softmax
  t0 <- getCPUTime
  s <- evaluate (sum (softmax x))
  t1 <- getCPUTime
  printf "  softmax row-wise                  %10.4f ms  sum=%s\n"
    (fromIntegral (t1-t0) / 1e9 :: Double) (show s)

  -- sdpa
  t2 <- getCPUTime
  s2 <- evaluate (sum (scaledDotProductAttention q k v mask))
  t3 <- getCPUTime
  printf "  sdpa (single head)                %10.4f ms  sum=%s\n"
    (fromIntegral (t3-t2) / 1e9 :: Double) (show s2)

  -- mha
  t4 <- getCPUTime
  s3 <- evaluate (sum (multiHeadAttention nHead x wq wk wv wo mask))
  t5 <- getCPUTime
  printf "  multi-head attention              %10.4f ms  sum=%s\n"
    (fromIntegral (t5-t4) / 1e9 :: Double) (show s3)

bench :: String -> Int -> Int -> IO () -> IO ()
bench name warmup iters action = do
  replicateM_ warmup action
  start <- getCPUTime
  replicateM_ iters action
  end <- getCPUTime
  let ns = end - start
      ms = fromIntegral ns / 1e9 :: Double
  printf "  %-37s %10.4f ms  (%5.1f µs each)\n"
    name ms (ms * 1000.0 / fromIntegral iters)
