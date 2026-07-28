{-# LANGUAGE OverloadedStrings #-}

-- | Ladder oracles for 'Circuit.LLM.Mixer' (GPT-2 → linear → delta → gate).
--
-- Pedagogic imprint from the waterloo_intern worklog: fixed-capacity memory
-- needs an eviction policy; each step adds one.
module Main (main) where

import Circuit.LLM.Mixer
import Numeric.LinearAlgebra
  ( Matrix,
    norm_2,
    rows,
    scale,
    sumElements,
    toList,
    tr,
    (><),
    (<>),
  )
import Numeric.LinearAlgebra qualified as LA
import System.Exit (exitFailure)
import Text.Printf (printf)
import Prelude hiding ((<>))

assert :: String -> Bool -> IO ()
assert msg ok =
  if ok
    then putStrLn ("  PASS " ++ msg)
    else do
      putStrLn ("  FAIL " ++ msg)
      exitFailure

nearMat :: Double -> Matrix Double -> Matrix Double -> Bool
nearMat tol a b =
  rows a == rows b
    && LA.cols a == LA.cols b
    && norm_2 (LA.flatten (a - b)) < tol

-- | Deterministic toy Q,K,V: T×d
toyQKV :: Int -> Int -> (Matrix Double, Matrix Double, Matrix Double)
toyQKV t d =
  let q =
        (t >< d)
          [ sin (fromIntegral (i * 3 + j) * 0.2) * 0.5
          | i <- [0 .. t - 1],
            j <- [0 .. d - 1]
          ]
      k =
        (t >< d)
          [ cos (fromIntegral (i * 2 + j) * 0.15) * 0.5
          | i <- [0 .. t - 1],
            j <- [0 .. d - 1]
          ]
      v =
        (t >< d)
          [ fromIntegral (i + j) * 0.1
          | i <- [0 .. t - 1],
            j <- [0 .. d - 1]
          ]
   in (q, k, v)

main :: IO ()
main = do
  putStrLn "=== circuits-llm mixer ladder ==="
  let t = 8
      d = 4
      (q, k, v) = toyQKV t d

  -------------------------------------------------------------------------
  putStrLn "softmax + KV: step scan matches prefill"
  do
    let oFull = softmaxAttnPrefill q k v
        (oSteps, _) =
          foldl
            ( \(acc, cache) i ->
                let qi = LA.asRow (LA.toRows q !! i)
                    ki = LA.asRow (LA.toRows k !! i)
                    vi = LA.asRow (LA.toRows v !! i)
                    (oi, cache') = softmaxAttnStep qi ki vi cache
                 in (acc ++ [LA.flatten oi], cache')
            )
            ([], emptyKv d)
            [0 .. t - 1]
        oScan = LA.fromRows oSteps
    printf "  ||full - scan||_2 = %.3e\n" (norm_2 (LA.flatten (oFull - oScan)))
    assert "KV step scan ≡ prefill" (nearMat 1e-9 oFull oScan)

  -------------------------------------------------------------------------
  putStrLn "linear attention: scan shapes + finite"
  do
    let (o, st) = linearAttnPrefill q k v
    assert "linear out rows = T" (rows o == t)
    assert "linear S is d×d" (LA.size (linS st) == (d, d))
    assert "linear output finite" (all (not . isNaN) (toList (LA.flatten o)))

  -------------------------------------------------------------------------
  putStrLn "delta: write then read recovers v (unit key, β=1)"
  do
    -- k unit row, write v, read with same k as q
    let k0 = LA.asRow (LA.normalize (LA.fromList [1, 0, 0, 0]))
        v0 = LA.asRow (LA.fromList [0.5, -0.25, 0.1, 0.0])
        s0 = LA.konst 0 (d, d)
        (_, s1) = deltaAttnStep 1.0 k0 k0 v0 s0
        (o1, _) = deltaAttnStep 1.0 k0 k0 v0 s1
    -- After first write, S holds association; second step overwrites same key
    -- with delta relative to itself → o ≈ v on first read after write.
    -- First step: v_old=0, S = k^T v, o = k S = (k k^T) v = ||k||^2 v = v
    let (oWrite, sW) = deltaAttnStep 1.0 k0 k0 v0 s0
    printf "  ||o - v|| after first write = %.3e\n" (norm_2 (LA.flatten (oWrite - v0)))
    assert "delta first write: q=k recovers v" (nearMat 1e-9 oWrite v0)
    assert "delta S changed" (norm_2 (LA.flatten (sW - s0)) > 1e-9)
    -- silence unused
    assert "delta second step runs" (rows o1 == 1)

  -------------------------------------------------------------------------
  putStrLn "gated delta: α=0 forgets prior; α=1 matches pure delta"
  do
    let (oDelta, sDelta) = deltaAttnPrefill 0.8 q k v
        (oG1, sG1) = gatedDeltaPrefill 1.0 0.8 q k v
        (oG0, sG0) = gatedDeltaPrefill 0.0 0.8 q k v
    assert "α=1 gated ≡ pure delta out" (nearMat 1e-9 oG1 oDelta)
    assert "α=1 gated ≡ pure delta S" (nearMat 1e-9 sG1 sDelta)
    -- With α=0 each step forgets S before write; final S is only last write —
    -- not equal to pure delta S.
    assert "α=0 state differs from pure delta" (not (nearMat 1e-6 sG0 sDelta))
    assert "α=0 output finite" (all (not . isNaN) (toList (LA.flatten oG0)))

  -------------------------------------------------------------------------
  putStrLn "KDA-lite: per-channel α=1 matches gated α=1 step"
  do
    let q0 = LA.asRow (LA.toRows q !! 0)
        k0 = LA.asRow (LA.toRows k !! 0)
        v0 = LA.asRow (LA.toRows v !! 0)
        s0 = LA.konst 0 (d, d)
        alpha1 = LA.konst 1 d
        (oK, sK) = kdaLiteStep alpha1 0.9 q0 k0 v0 s0
        (oG, sG) = gatedDeltaStep 1.0 0.9 q0 k0 v0 s0
    assert "kda-lite α=1 ≡ gated α=1 out" (nearMat 1e-9 oK oG)
    assert "kda-lite α=1 ≡ gated α=1 S" (nearMat 1e-9 sK sG)

  -------------------------------------------------------------------------
  putStrLn "chunked additive linear (unnormalized) vs sequential outer sum"
  do
    -- Sequential unnormalized: S += k^T v, o_t = q_t S_t with φ = elu+1
    let qf = eluPlus1 q
        kf = eluPlus1 k
        seqUnnorm =
          let steps = zip3 (LA.toRows qf) (LA.toRows kf) (LA.toRows v)
              (outs, _) =
                foldl
                  ( \(acc, s) (qr, kr, vr) ->
                      let kR = LA.asRow kr
                          vR = LA.asRow vr
                          qR = LA.asRow qr
                          s' = s + (tr kR <> vR)
                          o = qR <> s'
                       in (acc ++ [LA.flatten o], s')
                  )
                  ([], LA.konst 0 (d, d))
                  steps
           in LA.fromRows outs
        -- Note: chunkedLinearAttn uses causal intra scores not pure recurrent
        -- outer product alone — equality is only for C=1 (pure recurrent).
        oC1 = chunkedLinearAttn 1 q k v
    printf "  ||chunk C=1 - seq||_2 = %.3e\n" (norm_2 (LA.flatten (oC1 - seqUnnorm)))
    assert "chunk C=1 ≡ sequential unnormalized linear" (nearMat 1e-8 oC1 seqUnnorm)
    let oC4 = chunkedLinearAttn 4 q k v
    assert "chunk C=4 shape T×d" (LA.size oC4 == (t, d))
    assert "chunk C=4 finite" (all (not . isNaN) (toList (LA.flatten oC4)))

  putStrLn "=== mixer ladder green ==="
