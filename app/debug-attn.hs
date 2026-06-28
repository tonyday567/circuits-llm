module Main where

import Control.Exception (evaluate)
import Data.Foldable (sum)
import Harpie.Array (Array, array, index, mult, shape, transpose)
import System.CPUTime (getCPUTime)
import Text.Printf (printf)

main :: IO ()
main = do
  let seqLen = 64; headDim = 16
  let q = array [seqLen, headDim] [1 | _ <- [1..seqLen*headDim]] :: Array Double
      k = q

  putStrLn "Testing mult..."
  t0 <- getCPUTime
  let scores = mult q (transpose k)
  t0b <- getCPUTime
  printf "  shape=%s time=%.3f ms\n" (show (shape scores)) (fromIntegral (t0b - t0) / 1e9 :: Double)

  -- Force first element
  t1 <- getCPUTime
  evaluate (index scores [0, 0]) >>= \v -> do
    t2 <- getCPUTime
    printf "  scores[0,0]=%s time=%.3f ms\n" (show v) (fromIntegral (t2-t1) / 1e9 :: Double)

  -- Force all elements via sum
  t3 <- getCPUTime
  evaluate (sum scores) >>= \s -> do
    t4 <- getCPUTime
    printf "  sum=%s time=%.3f ms\n" (show s) (fromIntegral (t4-t3) / 1e9 :: Double)

  putStrLn "DONE"
