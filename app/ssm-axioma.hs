module Main where

import Circuit.LLM.Attention (causalMask, multiHeadAttention)
import Circuit.LLM.SSM
import Circuit.Process (scan)
import Data.List (foldl', scanl')
import Data.Vector.Unboxed qualified as V
import Harpie.Array (Array, array, mult, shape, zipWith, (!))
import Prelude hiding (zipWith)

approx :: Double -> Double -> Bool
approx x y = abs (x - y) < 1e-9

approxArray :: Array Double -> Array Double -> Bool
approxArray x y =
  and (zipWith (\a b -> abs (a - b) < 1e-9) x y)

check :: String -> Bool -> IO Bool
check name ok = do
  putStrLn $ (if ok then "PASS " else "FAIL ") ++ name
  pure ok

-- | Hand-rolled EWMA for exact oracle comparison.
ewmaHand :: Double -> [Double] -> [Double]
ewmaHand alpha = tail . scanl' (\h x -> alpha * x + (1 - alpha) * h) 0

main :: IO ()
main = do
  let ewmaAsAff alpha x = Aff (1 - alpha) (alpha * x)
      steps = [Aff 0.5 1, Aff 0.5 2, Aff 0.5 3, Aff 0.5 4]
      ewmaSteps = map (ewmaAsAff 0.5) [1, 1, 1, 1]
      h0v = array [3] [0, 0, 0]
      vsteps =
        [ AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [1, 2, 3]),
          AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [4, 5, 6])
        ]
      -- Toy data for SSM vs attention coexistence demo.
      seqLen = 4
      nEmbd = 6
      nHead = 2
      headDim = nEmbd `div` nHead
      -- Input: [seqLen, nEmbd]
      embed = array [seqLen, nEmbd] $
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] ++
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] ++
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] ++
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      -- Identity-ish weight matrices for attention.
      wEye = array [nEmbd, nEmbd] $
        [ if i == j then 1.0 else 0.0
        | i <- [0 .. nEmbd - 1]
        , j <- [0 .. nEmbd - 1]
        ]
      mask4 = causalMask seqLen
      -- SSM layer: fixed diagonal A, B = input embedding row.
      ssmA = array [nEmbd] (replicate nEmbd 0.5)
      ssmH0 = array [nEmbd] (replicate nEmbd 0.0)
      -- Convert each row of embed to an AffVec.
      toRows :: Array Double -> [Array Double]
      toRows x =
        let [n, d] = V.toList (shape x)
         in [ array [d] [x ! [r, c] | c <- [0 .. d - 1]] | r <- [0 .. n - 1] ]
      embedRows = toRows embed
      affVecs = [AffVec ssmA row | row <- embedRows]
  results <-
    sequence
      [ check "SSM affComp doctest matches" $
          let Aff a b = affComp (Aff 5 7) (Aff 2 3)
           in approx a 10 && approx b 22,
        check "SSM sequential scan matches hand" $
          seqSSM 0 [Aff 1 1, Aff 1 2, Aff 1 3] == [1, 3, 6],
        check "SSM ewma 1x1 equals hand EWMA (exact)" $
          let alpha = 0.6
              inputs = [1 .. 10] :: [Double]
              ssmResult = seqSSM 0 [Aff (1 - alpha) (alpha * x) | x <- inputs]
              ewmaResult = ewmaHand alpha inputs
           in and [approx x y | (x, y) <- zip ssmResult ewmaResult],
        check "SSM ewma 1x1 equals hand EWMA (alpha=0.2)" $
          let alpha = 0.2
              inputs = [5, -3, 8, 2] :: [Double]
              ssmResult = seqSSM 0 [Aff (1 - alpha) (alpha * x) | x <- inputs]
              ewmaResult = ewmaHand alpha inputs
           in and [approx x y | (x, y) <- zip ssmResult ewmaResult],
        check "SSM associative scan equals sequential scan (constant A)" $
          let seqResult = seqSSM 0 ewmaSteps
              assocResult = assocSSM 0 ewmaSteps
           in and [approx x y | (x, y) <- zip seqResult assocResult],
        check "SSM associative scan equals sequential scan (varying A)" $
          let seqResult = seqSSM 0 steps
              assocResult = assocSSM 0 steps
           in and [approx x y | (x, y) <- zip seqResult assocResult],
        check "SSM Process scan equals sequential scan" $
          let procResult = scan ssmProcess steps
              seqResult = seqSSM 0 steps
           in and [approx x y | (x, y) <- zip procResult seqResult],
        check "SSM Process scan equals associative scan" $
          let procResult = scan ssmProcess steps
              assocResult = assocSSM 0 steps
           in and [approx x y | (x, y) <- zip procResult assocResult],
        check "SSM vector sequential scan equals associative scan" $
          let seqResult = seqSSMVec h0v vsteps
              assocResult = assocSSMVec h0v vsteps
           in and [approxArray x y | (x, y) <- zip seqResult assocResult],
        check "SSM vector matches scalar on dim-1 slices" $
          let h0scalar = array [1] [0]
              scalarSteps =
                [ AffVec (array [1] [0.5]) (array [1] [1]),
                  AffVec (array [1] [0.5]) (array [1] [2])
                ]
              seqResult = map (\a -> a ! [0]) (seqSSMVec h0scalar scalarSteps)
              assocResult = assocSSM 0 [Aff 0.5 1, Aff 0.5 2]
           in and [approx x y | (x, y) <- zip seqResult assocResult],
        check "SSM System view typechecks" $
          let (outs, _sF) = runSystem ssmSystemVec h0v vsteps
           in length outs == length vsteps,
        check "SSM System scan equals sequential scan" $
          let (sysResult, _sF) = runSystem ssmSystemVec h0v vsteps
              seqResult = seqSSMVec h0v vsteps
           in and [approxArray x y | (x, y) <- zip sysResult seqResult],
        check "SSM vs attention: same-shape output on shared embed" $
          let ssmOut = seqSSMVec ssmH0 affVecs
              attnOut =
                multiHeadAttention nHead embed wEye wEye wEye wEye mask4
              -- SSM: one output per step; attention: one output for full sequence
              ssmRows = length ssmOut
              [attnRows, attnCols] = V.toList (shape attnOut)
           in ssmRows == seqLen
                && attnRows == seqLen
                && attnCols == nEmbd
                && length ssmOut == length affVecs
      ]
  if and results
    then putStrLn "\nAll tests passed."
    else error "Some tests failed."
