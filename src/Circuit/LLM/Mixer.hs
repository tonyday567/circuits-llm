{-# LANGUAGE OverloadedStrings #-}

-- | Stateful token mixers from the pedagogic GPT-2 → linear → delta ladder
-- (waterloo_intern worklog / Kimi lineage).
--
-- Already in this package: full-sequence softmax MHA ('Circuit.LLM.GPT',
-- 'Circuit.LLM.Attention').  This module adds the /memory/ progression:
--
-- * softmax decode with an explicit KV cache
-- * linear attention with fixed-size state @S@ (and normalizer @z@)
-- * delta-rule write (overwrite what the key already stores)
-- * gated decay of @S@ (Mamba-style scalar, or per-channel vector for KDA-lite)
-- * chunked additive linear scan (matches sequential; training-shaped)
--
-- Shapes (single head, teaching form):
--
-- @
-- q, k, v  :: Matrix  -- T×d  (rows = time, cols = head dim)
-- S        :: Matrix  -- d×d  associative memory
-- @
--
-- Multi-head is a 'map' over heads later; keep one head honest first.
--
-- Pedagogic imprint: same blocks as the article, hmatrix BLAS, pure scans —
-- the discrete Process / decaying Stats story, but for attention state.
module Circuit.LLM.Mixer
  ( -- * Softmax + KV cache
    KvCache (..),
    emptyKv,
    softmaxAttnPrefill,
    softmaxAttnStep,

    -- * Linear attention (fixed state)
    LinearState (..),
    emptyLinear,
    eluPlus1,
    linearAttnPrefill,
    linearAttnStep,

    -- * Delta rule
    deltaAttnStep,
    deltaAttnPrefill,

    -- * Gated memory
    gatedDeltaStep,
    gatedDeltaPrefill,
    kdaLiteStep,

    -- * Chunked additive linear (matches sequential)
    chunkedLinearAttn,

    -- * Scans
    scanMixer,
  )
where

import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromLists,
    fromRows,
    konst,
    rows,
    scale,
    sumElements,
    toList,
    toRows,
    tr,
    (<>),
  )
import Numeric.LinearAlgebra qualified as LA
import Prelude hiding ((<>))

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

-- | Causal upper-triangle mask: 0 on/below diagonal, -∞ above.
causalMask :: Int -> Matrix Double
causalMask n =
  fromLists
    [ [if j <= i then 0 else -(1 / 0) | j <- [0 .. n - 1]]
    | i <- [0 .. n - 1]
    ]

-- | Row-wise stable softmax.
softmaxRows :: Matrix Double -> Matrix Double
softmaxRows x = fromRows [soft (toList r) | r <- toRows x]
  where
    soft xs =
      let m = maximum xs
          e = map (\v -> exp (v - m)) xs
          s = sum e
       in LA.fromList (map (/ s) e)

-- | Scaled dot-product attention (full sequence), causal.
softmaxAttnPrefill :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
softmaxAttnPrefill q k v =
  let t = rows q
      dk = fromIntegral (cols k) :: Double
      scores = scale (1 / sqrt dk) (q <> tr k) + causalMask t
      attn = softmaxRows scores
   in attn <> v

----------------------------------------------------------------------
-- Softmax + KV cache
----------------------------------------------------------------------

-- | Growing key/value cache for decoder-style softmax attention.
data KvCache = KvCache
  { kvK :: Matrix Double, -- T_past × d
    kvV :: Matrix Double
  }
  deriving (Eq, Show)

-- | Empty cache (0 rows).
emptyKv :: Int -> KvCache
emptyKv d = KvCache (konst 0 (0, d)) (konst 0 (0, d))

-- | One decode step: @q,k,v@ are 1×d (current token). Returns output 1×d and
-- extended cache.
softmaxAttnStep ::
  Matrix Double -> -- q 1×d
  Matrix Double -> -- k 1×d
  Matrix Double -> -- v 1×d
  KvCache ->
  (Matrix Double, KvCache)
softmaxAttnStep q k v cache =
  let kAll = if rows (kvK cache) == 0 then k else kvK cache LA.=== k
      vAll = if rows (kvV cache) == 0 then v else kvV cache LA.=== v
      dk = fromIntegral (cols k) :: Double
      scores = scale (1 / sqrt dk) (q <> tr kAll) -- 1 × T
      attn = softmaxRows scores
      o = attn <> vAll
   in (o, KvCache kAll vAll)

----------------------------------------------------------------------
-- Linear attention (ELU+1 feature map, fixed S and z)
----------------------------------------------------------------------

-- | Fixed-size linear-attention memory: @S@ accumulates @φ(k)ᵀ v@ outer
-- products; @z@ accumulates @φ(k)@ for the normalizer.
data LinearState = LinearState
  { linS :: Matrix Double, -- d×d
    linZ :: Vector Double -- d
  }
  deriving (Eq, Show)

emptyLinear :: Int -> LinearState
emptyLinear d = LinearState (konst 0 (d, d)) (konst 0 d)

-- | Feature map φ(x) = ELU(x)+1 (article / Katharopoulos linear attention).
eluPlus1 :: Matrix Double -> Matrix Double
eluPlus1 = cmap (\x -> if x > 0 then x + 1 else exp x)

-- | One token of linear attention. @q,k,v@ are 1×d.
--
-- @
-- S ← S + φ(k)ᵀ v
-- z ← z + φ(k)
-- o  = (φ(q) S) / (φ(q)·z)
-- @
linearAttnStep ::
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  LinearState ->
  (Matrix Double, LinearState)
linearAttnStep q k v st =
  let qf = eluPlus1 q -- 1×d
      kf = eluPlus1 k
      -- S is d×d storing sum_i φ(k_i)^T v_i  (outer: d×1 * 1×d)
      kCol = tr kf -- d×1
      vRow = v -- 1×d
      s' = linS st + (kCol <> vRow)
      z' = linZ st + LA.flatten kf
      -- o = φ(q) @ S  → 1×d; denom = φ(q)·z
      num = qf <> s' -- 1×d
      denom = sumElements (qf * LA.asRow z') -- scalar as sum of elementwise
      o =
        if abs denom < 1e-12
          then konst 0 (1, cols q)
          else scale (1 / denom) num
   in (o, LinearState s' z')

-- | Prefill by scanning tokens (rows of q,k,v).
linearAttnPrefill ::
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, LinearState)
linearAttnPrefill q k v =
  let d = cols q
      steps = zip3 (toRows q) (toRows k) (toRows v)
      (outs, st) =
        foldl'
          ( \(acc, s) (qr, kr, vr) ->
              let (o, s') =
                    linearAttnStep
                      (LA.asRow qr)
                      (LA.asRow kr)
                      (LA.asRow vr)
                      s
               in (acc ++ [LA.flatten o], s')
          )
          ([], emptyLinear d)
          steps
   in (fromRows outs, st)

----------------------------------------------------------------------
-- Delta rule
----------------------------------------------------------------------

-- | One delta-rule step (sequential form from the article).
--
-- @
-- v_old = k S          -- what this key already retrieves (k is 1×d, S is d×d:
--                      -- we use S as mapping d→d on row vectors: o = q <> S)
-- u     = β (v - v_old)
-- S'    = S + kᵀ u
-- o     = q S'
-- @
--
-- Convention: treat @S@ as right-acting on 1×d queries (@o = q <> S@), so
-- writes are @S += kᵀ <> u@ with @k,u@ as 1×d.
deltaAttnStep ::
  Double -> -- β write strength in (0,1] typically sigmoid
  Matrix Double -> -- q 1×d
  Matrix Double -> -- k 1×d
  Matrix Double -> -- v 1×d
  Matrix Double -> -- S d×d
  (Matrix Double, Matrix Double)
deltaAttnStep beta q k v s =
  let vOld = k <> s -- 1×d
      u = scale beta (v - vOld)
      s' = s + (tr k <> u) -- d×d
      o = q <> s'
   in (o, s')

-- | Prefill with constant β.
deltaAttnPrefill ::
  Double ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, Matrix Double)
deltaAttnPrefill beta q k v =
  let d = cols q
      steps = zip3 (toRows q) (toRows k) (toRows v)
      (outs, sF) =
        foldl'
          ( \(acc, s) (qr, kr, vr) ->
              let (o, s') =
                    deltaAttnStep
                      beta
                      (LA.asRow qr)
                      (LA.asRow kr)
                      (LA.asRow vr)
                      s
               in (acc ++ [LA.flatten o], s')
          )
          ([], konst 0 (d, d))
          steps
   in (fromRows outs, sF)

----------------------------------------------------------------------
-- Gated delta (scalar α) and KDA-lite (per-channel α)
----------------------------------------------------------------------

-- | Gated delta step: decay previous state, then delta-write.
--
-- @
-- S ← α S
-- then deltaAttnStep β
-- @
--
-- @α = 1@ → pure delta; @α = 0@ → forget all then write.
gatedDeltaStep ::
  Double -> -- α ∈ [0,1]
  Double -> -- β
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, Matrix Double)
gatedDeltaStep alpha beta q k v s =
  deltaAttnStep beta q k v (scale alpha s)

gatedDeltaPrefill ::
  Double ->
  Double ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, Matrix Double)
gatedDeltaPrefill alpha beta q k v =
  let d = cols q
      steps = zip3 (toRows q) (toRows k) (toRows v)
      (outs, sF) =
        foldl'
          ( \(acc, s) (qr, kr, vr) ->
              let (o, s') =
                    gatedDeltaStep
                      alpha
                      beta
                      (LA.asRow qr)
                      (LA.asRow kr)
                      (LA.asRow vr)
                      s
               in (acc ++ [LA.flatten o], s')
          )
          ([], konst 0 (d, d))
          steps
   in (fromRows outs, sF)

-- | KDA-lite: per-channel decay @α :: Vector d@ (elementwise scale of each
-- column of @S@ before the delta write). Full KDA has richer parameterisation;
-- this imprints the /fine-grained forget/ idea.
kdaLiteStep ::
  Vector Double -> -- α length d
  Double -> -- β
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, Matrix Double)
kdaLiteStep alpha beta q k v s =
  let a = toList alpha
      sDecayed =
        fromLists
          [ [ sij * (a !! j)
            | (j, sij) <- zip [0 ..] row
            ]
          | row <- LA.toLists s
          ]
   in deltaAttnStep beta q k v sDecayed

----------------------------------------------------------------------
-- Chunked additive linear (training-shaped; matches sequential)
----------------------------------------------------------------------

-- | Chunked additive linear attention (article intermediate form).
-- Within chunk: causal @q (kᵀ v)@-style via masked scores on φ features;
-- across chunks: fold into @S@ and read @q S@.
--
-- For feature map = identity and no normalizer this is didactic; we use
-- φ = ELU+1 and still accumulate @S@ as in sequential linear (without @z@
-- in the chunk path for simplicity of the equality oracle — compare
-- unnormalized numerators).
chunkedLinearAttn ::
  Int -> -- chunk size C
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double -- T×d outputs (unnormalized φ(q)S path)
chunkedLinearAttn cSize q k v =
  let t = rows q
      d = cols q
      qf = eluPlus1 q
      kf = eluPlus1 k
      nChunks = (t + cSize - 1) `div` cSize
      go i s acc
        | i >= nChunks = acc
        | otherwise =
            let lo = i * cSize
                hi = min t ((i + 1) * cSize)
                n = hi - lo
                qC = subRows lo n qf
                kC = subRows lo n kf
                vC = subRows lo n v
                -- inter: qC <> S  (n×d)
                oPrev = qC <> s
                -- intra: causal attn on features
                scores = (qC <> tr kC) * lowerTriOnes n
                oCurr = scores <> vC
                o = oPrev + oCurr
                s' = s + (tr kC <> vC)
             in go (i + 1) s' (acc ++ toRows o)
   in fromRows (go 0 (konst 0 (d, d)) [])

subRows :: Int -> Int -> Matrix Double -> Matrix Double
subRows lo n m = fromRows (take n (drop lo (toRows m)))

-- | Strict lower-triangular inclusive ones (causal 0/1 mask as multiply).
lowerTriOnes :: Int -> Matrix Double
lowerTriOnes n =
  fromLists
    [ [if j <= i then 1 else 0 | j <- [0 .. n - 1]]
    | i <- [0 .. n - 1]
    ]

----------------------------------------------------------------------
-- Generic scan
----------------------------------------------------------------------

-- | Scan a step @(q,k,v,state) → (o,state')@ over rows.
scanMixer ::
  (Matrix Double -> Matrix Double -> Matrix Double -> s -> (Matrix Double, s)) ->
  s ->
  Matrix Double ->
  Matrix Double ->
  Matrix Double ->
  (Matrix Double, s)
scanMixer step s0 q k v =
  let steps = zip3 (toRows q) (toRows k) (toRows v)
      (outs, sF) =
        foldl'
          ( \(acc, s) (qr, kr, vr) ->
              let (o, s') = step (LA.asRow qr) (LA.asRow kr) (LA.asRow vr) s
               in (acc ++ [LA.flatten o], s')
          )
          ([], s0)
          steps
   in (fromRows outs, sF)
