{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RebindableSyntax #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

-- | AdamW-as-metric regression test and micro-benchmark.
--
-- The standard AdamW update @param -= lr_t * m / (sqrt v + eps) + wd_lr * param@
-- is re-expressed as applying the inverse metric @g^-1 = diag(1 / (sqrt v + eps))@
-- to the first-moment cotangent @m@, via 'raiseWith'.  The result is
-- oracle-checked against the hand-rolled 'Circuit.LLM.Training.updateVector'.
module Main where

import Circuit.AD.Metric (raiseWith)
import NumHask.Diff (Diff, pattern Diff)
import Circuit.LLM.Training (updateVector)
import Control.DeepSeq (NFData (..), force, ($!!))
import Data.Functor.Rep (liftR2)
import Control.Exception (evaluate)
import Data.List (zipWith3)
import Data.String (fromString)
import Data.Vector qualified as V
import GHC.TypeNats (KnownNat)
import Harpie.Fixed (Array (..), array, asVector)
import Harpie.NumHask ()
import Numeric.LinearAlgebra (Vector)
import Numeric.LinearAlgebra qualified as LA
import NumHask.Prelude hiding (fromList, sqrt, toList)
import NumHask.Prelude qualified as NH
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Text.Printf (printf)
import Prelude ()

-- | Orphan: 'Array' stores a boxed vector, so forcing it is forcing the vector.
instance (NFData a) => NFData (Array s a) where
  rnf (Array v) = rnf v

-- ----------------------------------------------------------------------
-- Metric preconditioners
-- ----------------------------------------------------------------------

-- | Adam's diagonal inverse metric, as a 'Diff' @Diff (v, cotangent) cotangent@.
--
-- Forward: @g^-1 v c = c / (sqrt v + eps)@.
adamMetric ::
  forall n.
  (KnownNat n) =>
  Double ->
  Diff (Array '[n] Double, Array '[n] Double) (Array '[n] Double)
adamMetric eps = Diff $ \(v, c) ->
  let lowered = liftR2 (\c_i v_i -> c_i / (NH.sqrt v_i + eps)) c v
   in ( lowered,
        \dc -> (zero, liftR2 (\dc_i v_i -> dc_i / (NH.sqrt v_i + eps)) dc v)
      )

-- | Identity metric @g = δ@; reduces the update to SGD with weight decay.
unitMetric ::
  forall n.
  (KnownNat n) =>
  Diff (Array '[n] Double, Array '[n] Double) (Array '[n] Double)
unitMetric = Diff $ \(_, c) -> (c, (zero,))

-- | Parameter update expressed through a metric 'Diff'.
updateWithMetric ::
  forall n.
  (KnownNat n) =>
  -- | inverse metric @g^-1@ as a 'Diff'
  Diff (Array '[n] Double, Array '[n] Double) (Array '[n] Double) ->
  Double ->
  Double ->
  Array '[n] Double ->
  Array '[n] Double ->
  Array '[n] Double ->
  Array '[n] Double
updateWithMetric g lr_t wd_lr param m v =
  let preconditioned = raiseWith g v m
   in param - preconditioned |* lr_t - param |* wd_lr

-- ----------------------------------------------------------------------
-- Conversion
-- ----------------------------------------------------------------------

toArray :: forall n. (KnownNat n) => Vector Double -> Array '[n] Double
toArray v = array (LA.toList v)

fromArray :: forall n. (KnownNat n) => Array '[n] Double -> Vector Double
fromArray a = LA.fromList (V.toList (asVector a))

-- ----------------------------------------------------------------------
-- Benchmark harness
-- ----------------------------------------------------------------------

bench :: (NFData a) => String -> IO a -> IO a
bench name action = do
  start <- getCurrentTime
  x <- action >>= evaluate . force
  end <- getCurrentTime
  let ms = realToFrac (diffUTCTime end start) * 1e3 :: Double
  printf "  %-40s %10.3f ms\n" name ms
  pure x

-- ----------------------------------------------------------------------
-- Main
-- ----------------------------------------------------------------------

runOracle :: IO ()
runOracle = do
  putStrLn "\n=== AdamW metric oracle (n=4) ==="
  let param = LA.fromList [1.0, 2.0, 3.0, 4.0] :: Vector Double
      m = LA.fromList [0.1, 0.2, 0.3, 0.4] :: Vector Double
      v = LA.fromList [0.01, 0.04, 0.09, 0.16] :: Vector Double
      lr_t = 0.01
      eps = 1e-8
      wd_lr = 0.001
      expected = updateVector lr_t eps wd_lr param m v
      actual =
        fromArray
          ( updateWithMetric @4 (adamMetric eps) lr_t wd_lr
              (toArray param)
              (toArray m)
              (toArray v)
          )
  printf "  expected: %s\n" (show (LA.toList expected))
  printf "  actual:   %s\n" (show (LA.toList actual))
  if all (\(e, a) -> abs (e - a) < 1e-12) (zip (LA.toList expected) (LA.toList actual))
    then putStrLn "  PASS metric AdamW matches Training.updateVector"
    else error "FAIL metric AdamW diverges from Training.updateVector"

  putStrLn "\n=== Alternative g slots in (unit metric -> SGD+WD) ==="
  let sgdUpdate =
        fromArray
          ( updateWithMetric @4 unitMetric lr_t wd_lr
              (toArray param)
              (toArray m)
              (toArray v)
          )
      sgdByHand =
        LA.fromList $
          zipWith3 (\p m_ _v -> p - lr_t * m_ - wd_lr * p) (LA.toList param) (LA.toList m) (LA.toList v)
  printf "  sgd unit-metric: %s\n" (show (LA.toList sgdUpdate))
  printf "  sgd by hand:     %s\n" (show sgdByHand)
  if all (\(e, a) -> abs (e - a) < 1e-12) (zip (LA.toList sgdByHand) (LA.toList sgdUpdate))
    then putStrLn "  PASS unit metric reduces to SGD+WD"
    else error "FAIL unit metric does not reduce to SGD+WD"

runBenchmark :: IO ()
runBenchmark = do
  putStrLn "\n=== Micro-benchmark (n=10000, 1000 iters) ==="
  let n = 10000
      iters :: Int
      iters = 1000
      param0 = LA.fromList (take n (iterate (+ 0.0001) 1.0)) :: Vector Double
      m0 = LA.fromList (take n (iterate (+ 0.00001) 0.1)) :: Vector Double
      v0 = LA.fromList (take n (iterate (+ 0.000001) 0.01)) :: Vector Double
      lr_t = 0.001
      eps = 1e-8
      wd_lr = 0.001

  s1 <- bench "hand-rolled zipWith3" $ do
    let go (!p, !m_, !v_) _ = (updateVector lr_t eps wd_lr p m_ v_, m_, v_)
        (pFinal, _, _) = foldl' go (param0, m0, v0) [1 .. iters]
    pure (LA.sumElements pFinal)

  s2 <- bench "metric raiseWith (no conversion)" $ do
    let pA = toArray @10000 param0
        mA = toArray @10000 m0
        vA = toArray @10000 v0
        go (!p, !m_, !v_) _ = (updateWithMetric @10000 (adamMetric eps) lr_t wd_lr p m_ v_, m_, v_)
        (pFinal, _, _) = foldl' go (pA, mA, vA) [1 .. iters]
    pure (LA.sumElements (fromArray pFinal))

  s3 <- bench "metric raiseWith (with hmatrix conversion)" $ do
    let go (!p, !m_, !v_) _ =
          ( fromArray
              ( updateWithMetric @10000 (adamMetric eps) lr_t wd_lr
                  (toArray p)
                  (toArray m_)
                  (toArray v_)
              ),
            m_,
            v_
          )
        (pFinal, _, _) = foldl' go (param0, m0, v0) [1 .. iters]
    pure (LA.sumElements pFinal)

  putStrLn $ "  checksums: " ++ show (s1, s2, s3)

main :: IO ()
main = do
  runOracle
  runBenchmark
  putStrLn "\nAll checks passed."
