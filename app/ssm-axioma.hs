module Main where

import Circuit.LLM.SSM
import Circuit.Process (scan)
import Data.List (foldl')
import Data.Foldable (toList)
import Harpie.Array (Array, array, zipWith, (!))
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
  results <-
    sequence
      [ check "SSM affComp doctest matches" $
          let Aff a b = affComp (Aff 5 7) (Aff 2 3)
           in approx a 10 && approx b 22,
        check "SSM sequential scan matches hand" $
          seqSSM 0 [Aff 1 1, Aff 1 2, Aff 1 3] == [1, 3, 6],
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
           in and [approxArray x y | (x, y) <- zip sysResult seqResult]
      ]
  if and results
    then putStrLn "\nAll tests passed."
    else error "Some tests failed."
