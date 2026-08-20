{-# LANGUAGE OverloadedStrings #-}

module Main where

import Circuit.LLM.Backprop (crossEntropyBwd, geluBwd, linearBwd, softmaxBwd)
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromList,
    fromRows,
    konst,
    maxElement,
    reshape,
    rows,
    scale,
    sumElements,
    toList,
  )
import qualified Numeric.LinearAlgebra as LA

main :: IO ()
main = do
  -- Test 1: cross-entropy backward
  putStrLn "=== Cross-entropy backward ==="
  let logits = fromRows [fromList [0.1, 0.2, 0.7 :: Double], fromList [0.3, 0.5, 0.2]]
      targets = [2, 1]
      (loss0, grad0) = crossEntropyBwd logits targets
  putStrLn $ "  loss = " ++ show loss0
  putStrLn $ "  grad sum = " ++ show (sumElements grad0)
  putStrLn $ if abs (sumElements grad0) < 1e-10 then "  PASS (grad sums to 0)" else "  FAIL"

  -- Test 2: softmax backward
  putStrLn "\n=== Softmax backward ==="
  let scores = fromRows [fromList [1.0, 2.0, 3.0 :: Double]]
      probs = softmaxScores scores
      gradIn = softmaxBwd probs (reshape 3 (fromList (replicate 3 1.0)))
  case LA.toRows probs of
    (row : _) -> putStrLn $ "  probs = " ++ show (toList row)
    [] -> putStrLn "  probs = []"
  putStrLn $ "  grad sum = " ++ show (sumElements gradIn)
  putStrLn $ if abs (sumElements gradIn) < 1e-10 then "  PASS (grad sums to 0)" else "  FAIL"

  -- Test 3: linear backward
  putStrLn "\n=== Linear backward ==="
  let x = reshape 2 (fromList [1, 2, 3, 4 :: Double]) -- [2x2]
      w = reshape 2 (fromList [1, 0, 0, 1 :: Double]) -- [2x2] identity
      y = x LA.<> w
      ones = reshape (cols y) (fromList (replicate (rows y * cols y) 1.0))
      (gradX, gradW, _) = linearBwd x w ones
  putStrLn "  x = [1,2; 3,4], W = I, dL/dy = ones"
  putStrLn $ "  gradX = " ++ show (toList (LA.flatten gradX))
  putStrLn $ "  gradW = " ++ show (toList (LA.flatten gradW))
  -- gradX should be gradY @ W^T = ones @ I = ones
  let expectedGradX = replicate 4 1
  putStrLn $ if toList (LA.flatten gradX) == expectedGradX then "  PASS" else "  FAIL"

  -- Test 4: GELU backward
  putStrLn "\n=== GELU backward ==="
  let gx = reshape 1 (fromList [-2, -1, 0, 1, 2 :: Double])
      gy = reshape 1 (fromList (replicate 5 1.0))
      gxBwd = geluBwd gx gy
      gAnalytic = toList (LA.flatten gxBwd)
  putStrLn $ "  grad at x=[-2,-1,0,1,2] = " ++ show (take 3 gAnalytic) ++ "..."
  -- GELU at 0: 0 * phi'(0) + phi(0) = 0 + 0.5 = 0.5
  putStrLn $ if abs (gAnalytic !! 2 - 0.5) < 0.1 then "  PASS" else "  FAIL"

  putStrLn "\nAll gradient checks complete."

softmaxScores :: Matrix Double -> Matrix Double
softmaxScores m =
  fromRows
    [ let mx = maxElement row
          shifted = cmap (\x -> exp (x - mx)) row
          sm = sumElements shifted
       in scale (1 / sm) shifted
    | row <- LA.toRows m
    ]
