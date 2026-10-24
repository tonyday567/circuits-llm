-- High-margin probe for the Softmax Collapse bug (Barron et al. 2026,
-- arXiv:2610.31642) and its fix in crossEntropyBwd.
--
-- Three gradient forms are compared per margin:
--   textbook ⟜ the old softmax-minus-onehot form, kept inline as the
--     negative control; past the fp64 cliff (margin ~36.7) its target
--     gradient rounds to exactly zero while the others do not.
--   library  ⟜ crossEntropyBwd from Circuit.LLM.Backprop (the ported fix:
--     loss = softplus(-m), margin form, gradients in closed form).
--   reference ⟜ Barron's closed form written inline, an independent
--     transcription for cross-checking the port.
--
-- Run from the repo root: runghc smoke/smoke-ce-margin.hs

module Main where

import Circuit.LLM.Backprop (crossEntropyBwd, maskedCrossEntropyBwd)
import Control.Monad (forM)
import GHC.Float (log1p)
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    fromList,
    fromLists,
    sumElements,
    toList,
    toRows,
  )
import System.Exit (ExitCode (..), exitWith)

vocab :: Int
vocab = 5

target :: Int
target = 2

-- logit row: target at height m, others at zero
logitsAt :: Double -> Matrix Double
logitsAt m = fromLists [[if i == target then m else 0 | i <- [0 .. vocab - 1]]]

theRow :: Matrix Double -> Vector Double
theRow x = case toRows x of
  (r : _) -> r
  [] -> error "smoke-ce-margin: empty matrix"

softmaxStableL :: [Double] -> [Double]
softmaxStableL zv =
  let mx = maximum zv
      shifted = map (\x -> exp (x - mx)) zv
      s = sum shifted
   in map (/ s) shifted

-- the pre-fix textbook gradient: p_i - 1[i == t]
textbookRow :: [Double] -> Int -> [Double]
textbookRow zv t =
  let pv = softmaxStableL zv
   in [if i == t then pv !! i - 1 else pv !! i | i <- [0 .. length pv - 1]]

-- Barron's fixed loss and gradient for one row with hard target t.
-- m = z_t - LSE(others); loss = softplus(-m);
-- g_j = exp(z_j - z_t - loss) for j /= t; g_t = -sum others.
barronRow :: [Double] -> Int -> (Double, [Double])
barronRow zv t =
  let n = length zv
      others = [j | j <- [0 .. n - 1], j /= t]
      lseOthers = logSumExp [zv !! j | j <- others]
      m = (zv !! t) - lseOthers
      loss = softplus (-m)
      gOthers = [if j == t then 0 else exp ((zv !! j) - (zv !! t) - loss) | j <- [0 .. n - 1]]
      gt = -sum gOthers
   in (loss, [if j == t then gt else gOthers !! j | j <- [0 .. n - 1]])
  where
    logSumExp xs =
      let mx = maximum xs
       in mx + log (sum [exp (x - mx) | x <- xs])
    softplus x =
      let ax = abs x
       in max x 0 + log1p (exp (-ax))

relClose :: Double -> Double -> Double -> Bool
relClose tol a b = abs (a - b) <= tol * max 1e-300 (max (abs a) (abs b))

-- central finite-difference gradient of the library loss at a row
finiteDiffRow :: ([Double] -> Double) -> [Double] -> [Double]
finiteDiffRow f zv =
  let h = 1e-6
      n = length zv
      fd i =
        let zp = [if j == i then x + h else x | (j, x) <- zip [0 ..] zv]
            zm = [if j == i then x - h else x | (j, x) <- zip [0 ..] zv]
         in (f zp - f zm) / (2 * h)
   in [fd i | i <- [0 .. n - 1]]

probe :: Double -> IO (Bool, String)
probe m = do
  let logits = logitsAt m
      zv = toList (theRow logits)
      (lossLib, gradLib) = crossEntropyBwd logits [target]
      gl = toList (theRow gradLib)
      gt = textbookRow zv target
      (lossBar, gb) = barronRow zv target
      sLib = sumElements gradLib
      sBar = sum gb
      sTxt = sum gt
      agreesBarron = and [relClose 1e-9 (gl !! j) (gb !! j) | j <- [0 .. vocab - 1]] && relClose 1e-9 lossLib lossBar
      libSumsZero = relClose 1e-12 sLib 0
      libTargetNonZero = gl !! target /= 0
      -- negative control: the textbook form must still show the fingerprint past the cliff
      collapsedRegime = m >= 40
      bugShown = (gt !! target == 0) && sTxt > 0
      -- loss/gradient coherence of the port in the normal regime
      normalRegime = m <= 2
      fdg = finiteDiffRow (fst . barronLoss) zv
      libMatchesFD = and [relClose 1e-5 (gl !! j) (fdg !! j) | j <- [0 .. vocab - 1]]
      barronLoss zl = barronRow zl target
      -- masked variants: all-supervised must equal the library; all-masked must be zero
      (lossMask, gradMask) = maskedCrossEntropyBwd logits [target] [True]
      (lossNone, gradNone) = maskedCrossEntropyBwd logits [target] [False]
      maskSupervisedOk = lossMask == lossLib && sumElements gradMask == sLib
      maskNoneOk = lossNone == 0 && sumElements gradNone == 0
      pass =
        agreesBarron
          && libSumsZero
          && libTargetNonZero
          && (not collapsedRegime || bugShown)
          && (not normalRegime || libMatchesFD)
          && maskSupervisedOk
          && maskNoneOk
      report =
        "margin "
          ++ show m
          ++ "\n"
          ++ "  textbook: g_t "
          ++ show (gt !! target)
          ++ "  sum g "
          ++ show sTxt
          ++ "\n"
          ++ "  library:  loss "
          ++ show lossLib
          ++ "  g_t "
          ++ show (gl !! target)
          ++ "  sum g "
          ++ show sLib
          ++ "\n"
          ++ "  barron:   loss "
          ++ show lossBar
          ++ "  g_t "
          ++ show (gb !! target)
          ++ "  sum g "
          ++ show sBar
          ++ "\n"
          ++ "  "
          ++ (if agreesBarron then "" else "[FAIL: library /= barron reference] ")
          ++ (if libSumsZero then "" else "[FAIL: library sum /= 0] ")
          ++ (if libTargetNonZero then "" else "[FAIL: library g_t == 0] ")
          ++ (if not collapsedRegime || bugShown then "" else "[FAIL: textbook fingerprint not shown] ")
          ++ (if not normalRegime || libMatchesFD then "" else "[FAIL: library gradient /= finite difference of loss] ")
          ++ (if maskSupervisedOk then "" else "[FAIL: masked (all supervised) /= library] ")
          ++ (if maskNoneOk then "" else "[FAIL: masked (all masked out) not zero] ")
  pure (pass, report)

main :: IO ()
main = do
  putStrLn "=== High-margin CE probe (Softmax Collapse fix, arXiv:2610.31642) ==="
  let margins = [1, 20, 36.7, 40, 80] :: [Double]
  rs <- forM margins probe
  mapM_ (\(_, r) -> putStrLn r) rs
  let allPass = all fst rs
  putStrLn (if allPass then "ALL PASS" else "FAILURES PRESENT")
  exitWith (if allPass then ExitSuccess else ExitFailure 1)
