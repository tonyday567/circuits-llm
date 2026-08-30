{-# LANGUAGE DerivingStrategies #-}

-- | Linear state-space model as a 'Process' and the associative-scan law.
--
-- A scalar linear SSM is the recurrence @h_t = a_t h_{t-1} + b_t@.  Each step
-- is an affine function @(a_t, b_t)@; composition of affine functions is
-- associative, so the whole sequence can be reduced via an associative scan
-- instead of a sequential fold.  This is the parallelisation principle behind
-- S4 / Mamba / linear attention.
module Circuit.LLM.SSM
  ( -- * Scalar affine SSM
    Aff (..),
    affComp,
    seqSSM,
    assocScan,
    assocSSM,
    chunkedScan,

    -- * Vector (harpie) affine SSM
    AffVec (..),
    affCompVec,
    seqSSMVec,
    assocScanVec,
    assocSSMVec,

    -- * Moore (,) view
    mooreMorphism,
    ssmSystemVec,

    -- * Multi-head Moore (,) view (Dirichlet tensor)
    multiHeadSSMSystem,
    runMultiHeadSSMSystem,
    runSharedInputMultiHeadSSMSystem,

    -- * Process / Moore (,) view
    ssmProcess,
    ssmSystem,

    -- * Centrality pair (multi-head coupling)
    coupledMultiHeadSSMSystem,
  )
where

import Circuit.Body (Body (..))
import Circuit.Poly (Mono, Poly (PTensor))
import Circuit.Process (Process (..), mooreAsProcess)
import Circuit.Moore (Moore (..), monoDir, monoIn, mooreMachine, moore)
import Data.List (foldl1', scanl')
import Data.Void (absurd)
import Harpie.Array (Array, zipWith)
import Prelude hiding (zipWith)

-- | An affine function @h -> a * h + b@.
data Aff = Aff
  { affA :: Double,
    affB :: Double
  }
  deriving stock (Eq, Show)

-- | Composition of affine functions: @(f2 . f1)(h) = f2(f1(h))@.
--
-- >>> let f1 = Aff 2 3; f2 = Aff 5 7 in affComp f2 f1
-- Aff {affA = 10.0, affB = 22.0}
--
-- Because @(f2 . f1)(h) = 5*(2*h+3)+7 = 10*h + 22@.
affComp :: Aff -> Aff -> Aff
affComp (Aff a2 b2) (Aff a1 b1) = Aff (a2 * a1) (a2 * b1 + b2)

-- | Sequential scan of a scalar SSM from an initial state.
--
-- >>> seqSSM 0 [Aff 1 1, Aff 1 2, Aff 1 3]
-- [1.0,3.0,6.0]
seqSSM :: Double -> [Aff] -> [Double]
seqSSM h0 affs = case scanl' step h0 affs of
  (_ : outs) -> outs
  [] -> []
  where
    step h (Aff a b) = a * h + b

-- | Associative scan on affine coefficients.  Returns the list of composed
-- affine functions @[f1, f2 . f1, f3 . f2 . f1, ...]@.
--
-- The operator is reversed composition because 'scanl'' threads the accumulator
-- on the left, but we want the rightmost function applied first.
assocScan :: [Aff] -> [Aff]
assocScan affs = case scanl' (flip affComp) (Aff 1 0) affs of
  (_ : comps) -> comps
  [] -> []

-- | Tree-shaped associative scan with a given chunk size.
--
-- Each chunk is reduced to a summary affine function, the summaries are
-- prefix-composed, and the result is expanded back to a per-step prefix.  If
-- 'affComp' is associative, this must agree with 'assocScan' for every chunk
-- size.  This is the genuine parallelisation principle behind S4 / Mamba /
-- linear attention.
chunkedScan :: Int -> [Aff] -> [Aff]
chunkedScan _ [] = []
chunkedScan k affs
  | k <= 1 = assocScan affs
  | otherwise =
      let chunks = chunksOf k affs
          summaries = map (foldl1' (flip affComp)) chunks
          summaryPrefixes = assocScan summaries
          expand chunk prevPref =
            let inner = case scanl' (flip affComp) (Aff 1 0) chunk of (_ : xs) -> xs; [] -> []
             in map (\x -> affComp x prevPref) inner
       in concat [expand chunk prevPref | (chunk, prevPref) <- zip chunks (Aff 1 0 : summaryPrefixes)]
  where
    chunksOf _ [] = []
    chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- | Apply the associatively-scanned affine functions to an initial state.
assocSSM :: Double -> [Aff] -> [Double]
assocSSM h0 = map (\(Aff a b) -> a * h0 + b) . assocScan

-- | A 'Moore (,)' whose state is the hidden state @h@ and whose output is @h@.
-- Input is the affine coefficient pair @(a_t, b_t)@; the initial state @h0@ is
-- supplied when converting to a 'Process' or running directly.
ssmSystem :: Moore (,) (->) Double (Mono Aff Double)
ssmSystem = mooreMachine step extract
  where
    step h (Aff a b) = a * h + b
    extract h = h

-- | A 'Process' whose state is the hidden state @h@ and whose output is @h@.
-- Input is the affine coefficient pair @(a_t, b_t)@.
--
-- This is the first-input-seeded presentation with @h0 = 0@.  Use
-- 'ssmSystem' with 'mooreAsProcess' when you need a non-zero seed.
ssmProcess :: Process Aff Double
ssmProcess = mooreAsProcess ssmSystem 0

-- ---------------------------------------------------------------------------
-- Vector (harpie) affine SSM — diagonal-matrix state
-- ---------------------------------------------------------------------------

-- | Elementwise affine function on a harpie 1-D array:
-- @h_i -> a_i * h_i + b_i@.  This is the diagonal-matrix special case of the
-- full matrix SSM; it already exercises the associative-scan law on arrays.
data AffVec = AffVec
  { affAVec :: Array Double,
    affBVec :: Array Double
  }
  deriving stock (Eq, Show)

-- | Elementwise composition of diagonal affine functions.
affCompVec :: AffVec -> AffVec -> AffVec
affCompVec (AffVec a2 b2) (AffVec a1 b1) =
  AffVec (zipWith (*) a2 a1) (zipWith (+) (zipWith (*) a2 b1) b2)

-- | Sequential scan of a vector SSM.
seqSSMVec :: Array Double -> [AffVec] -> [Array Double]
seqSSMVec h0 affs = case scanl' step h0 affs of
  (_ : outs) -> outs
  [] -> []
  where
    step h (AffVec a b) = zipWith (+) (zipWith (*) a h) b

-- | Associative scan on vector affine coefficients.  The shape is inherited
-- from the first input.
assocScanVec :: [AffVec] -> [AffVec]
assocScanVec [] = []
assocScanVec (firstStep : rest) =
  case scanl' (flip affCompVec) (AffVec ones zeros) (firstStep : rest) of
    (_ : comps) -> comps
    [] -> []
  where
    ones = zipWith (\_ _ -> 1) (affAVec firstStep) (affAVec firstStep)
    zeros = zipWith (\_ _ -> 0) (affBVec firstStep) (affBVec firstStep)

-- | Apply the associatively-scanned vector affine functions to an initial state.
assocSSMVec :: Array Double -> [AffVec] -> [Array Double]
assocSSMVec h0 = map (\(AffVec a b) -> zipWith (+) (zipWith (*) a h0) b) . assocScanVec

-- ---------------------------------------------------------------------------
-- Moore (,) view: linear SSM as a Moore machine
-- ---------------------------------------------------------------------------

-- | Run a deterministic 'Moore (,)' with a monomial interface over a list of
-- inputs.  This is the same semantics as 'Circuit.Process.scan', but stated
-- directly on 'Moore (,)'.
mooreMorphism :: Moore (,) (->) s (Mono i o) -> s -> [i] -> ([o], s)
mooreMorphism (Moore (Body sys)) s0 is = go s0 is []
  where
    go s [] acc = (reverse acc, s)
    go s (i : iss) acc =
      let (s', (o, ())) = sys (s, monoIn i)
       in go s' iss (o : acc)

-- | Vector SSM as a 'Moore (,) (->)' with harpie state, input 'AffVec', and full
-- state observation.
ssmSystemVec :: Moore (,) (->) (Array Double) (Mono AffVec (Array Double))
ssmSystemVec = moore $ \(h, d) ->
  let AffVec a b = monoDir d
      h' = zipWith (+) (zipWith (*) a h) b
   in (h', (h', ()))

-- ---------------------------------------------------------------------------
-- Multi-head SSM as a PTensor-polynomial Moore (,)
-- ---------------------------------------------------------------------------

-- | Two independent vector SSM heads packaged as a single 'Moore (,)' over the
-- Dirichlet tensor @PTensor (Mono AffVec (Array Double)) (Mono AffVec (Array Double))@.
--
-- The two heads share the same input /direction type/ ('AffVec') but receive
-- independent direction values.  This is the right polynomial for parallel
-- layers: both heads fire on the same tick, each with its own input.  A
-- cartesian 'Prod' would force a choice between heads via @Either@ directions.
multiHeadSSMSystem ::
  Moore (,)
    (->)
    (Array Double, Array Double)
    (PTensor (Mono AffVec (Array Double)) (Mono AffVec (Array Double)))
multiHeadSSMSystem = moore $ \case
  ((h1, h2), (Right aff1, Right aff2)) ->
    let AffVec a1 b1 = aff1
        AffVec a2 b2 = aff2
        h1' = zipWith (+) (zipWith (*) a1 h1) b1
        h2' = zipWith (+) (zipWith (*) a2 h2) b2
     in ((h1', h2'), ((h1', ()), (h2', ())))
  (_, (Left v, _)) -> absurd v
  (_, (_, Left v)) -> absurd v

-- | Run the multi-head SSM with independent per-head inputs.
--
-- Returns the pair of head outputs at each step and the final pair of states.
runMultiHeadSSMSystem ::
  (Array Double, Array Double) ->
  [(AffVec, AffVec)] ->
  ([(Array Double, Array Double)], (Array Double, Array Double))
runMultiHeadSSMSystem s0 affPairs =
  let Moore (Body f) = multiHeadSSMSystem
      go s [] acc = (reverse acc, s)
      go (h1, h2) ((aff1, aff2) : affs') acc =
        let ((h1', h2'), ((o1, ()), (o2, ()))) = f ((h1, h2), (monoIn aff1, monoIn aff2))
         in go (h1', h2') affs' ((o1, o2) : acc)
   in go s0 affPairs []

-- | Run the multi-head SSM with the /same/ input supplied to both heads.
--
-- This is the diagonal shared-input case; building it from independent inputs
-- requires an explicit copy of the input direction.
runSharedInputMultiHeadSSMSystem ::
  (Array Double, Array Double) ->
  [AffVec] ->
  ([(Array Double, Array Double)], (Array Double, Array Double))
runSharedInputMultiHeadSSMSystem s0 affs = runMultiHeadSSMSystem s0 [(aff, aff) | aff <- affs]

-- | Two-headed SSM with a cross-head coupling: head 1 reads head 2's state.
--
-- This breaks premonoidal centrality: the order in which the two heads are
-- threaded through the shared medium matters, because head 1's update depends
-- on head 2's state.  It is the "flip" oracle for the multi-head centrality
-- pair.
coupledMultiHeadSSMSystem ::
  Moore (,)
    (->)
    (Array Double, Array Double)
    (PTensor (Mono AffVec (Array Double)) (Mono AffVec (Array Double)))
coupledMultiHeadSSMSystem = moore $ \case
  ((h1, h2), (Right aff1, Right aff2)) ->
    let AffVec a1 b1 = aff1
        AffVec a2 b2 = aff2
        -- Head 1 receives a cross-term from head 2's state.
        cross = zipWith (*) h2 (zipWith (\_ _ -> 0.1) h2 h2)
        h1' = zipWith (+) (zipWith (+) (zipWith (*) a1 h1) b1) cross
        h2' = zipWith (+) (zipWith (*) a2 h2) b2
     in ((h1', h2'), ((h1', ()), (h2', ())))
  (_, (Left v, _)) -> absurd v
  (_, (_, Left v)) -> absurd v
