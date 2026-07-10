{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Data.Foldable (sum)
import Harpie.Array (Array, array)
import Harpie.Array qualified as HA
import Harpie.Hmatrix (multM)
import Numeric.LinearAlgebra (Matrix, (<>))
import Numeric.LinearAlgebra qualified as LA
import System.CPUTime (getCPUTime)
import Text.Printf (printf)
import Prelude hiding (sum, (<>))

randArray :: Int -> Int -> Double -> Array Double
randArray rows cols seed = array [rows, cols] (take (rows * cols) values)
  where
    values = iterate nextR (sin seed * 0.5 + 0.5)
    nextR x = sin (x * 127.1 + 311.7) * 0.5 + 0.5

randMatrix :: Int -> Int -> Double -> Matrix Double
randMatrix r c seed = LA.reshape c (LA.fromList (take (r * c) values))
  where
    values = iterate nextR (sin seed * 0.5 + 0.5)
    nextR x = sin (x * 127.1 + 311.7) * 0.5 + 0.5

bench :: String -> [(a, b)] -> ((a, b) -> Double) -> IO ()
bench name inputs action = do
  let runOne = evaluate . force . action
  mapM_ runOne inputs -- warmup and force
  start <- getCPUTime
  mapM_ runOne inputs
  end <- getCPUTime
  let iters = length inputs
      ms = fromIntegral (end - start) / 1e9 :: Double
      us = ms * 1000.0 / fromIntegral iters
  printf "  %-30s %8.3f ms  (%8.2f us each)\n" name ms us

main :: IO ()
main = do
  putStrLn "=== Harpie generic vs BLAS fast path ==="

  let configs :: [(Int, Int)]
      configs = [(16, 1000), (64, 100), (256, 10), (512, 5)]

  mapM_
    ( \(n, iters) -> do
        let inputsHarpie = [(randArray n n (fromIntegral i), randArray n n (fromIntegral i + 1000)) | i <- [1 .. iters]]
            inputsHmatrix = [(randMatrix n n (fromIntegral i), randMatrix n n (fromIntegral i + 1000)) | i <- [1 .. iters]]
        printf "\n%dx%d matrix multiply (%d iters):\n" n n iters
        bench "harpie generic mult" inputsHarpie (\(a, b) -> sum (a `HA.mult` b))
        bench "harpie hmatrix multM" inputsHarpie (\(a, b) -> sum (a `multM` b))
        bench "hmatrix direct" inputsHmatrix (\(a, b) -> LA.sumElements (a <> b))
    )
    configs

  putStrLn "\nDONE"
