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

    -- * Vector (harpie) affine SSM
    AffVec (..),
    affCompVec,
    seqSSMVec,
    assocScanVec,
    assocSSMVec,

    -- * System view
    runSystem,
    ssmSystemVec,

    -- * Process view
    ssmProcess,
  )
where

import Circuit.Poly (Mono, System (..), monoDir, monoIn)
import Circuit.Process (Process (..))
import Data.List (foldl', scanl')
import Harpie.Array (Array, array, zipWith)
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

-- | Apply the associatively-scanned affine functions to an initial state.
assocSSM :: Double -> [Aff] -> [Double]
assocSSM h0 = map (\(Aff a b) -> a * h0 + b) . assocScan

-- | A 'Process' whose state is the hidden state @h@ and whose output is @h@.
-- Input is the affine coefficient pair @(a_t, b_t)@.
ssmProcess :: Process Aff Double
ssmProcess = Process inject step extract
  where
    inject (Aff a b) = a * 0 + b
    step h (Aff a b) = a * h + b
    extract h = h


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
-- System view: linear SSM as a Moore machine
-- ---------------------------------------------------------------------------

-- | Run a deterministic 'System' with a monomial interface over a list of
-- inputs.  This is the same semantics as 'Circuit.Process.scan', but stated
-- directly on 'System'.
runSystem :: System (->) s (Mono i o) -> s -> [i] -> ([o], s)
runSystem (System sys) s0 is = go s0 is []
  where
    go s [] acc = (reverse acc, s)
    go s (i : iss) acc =
      let (s', (o, ())) = sys (s, monoIn i)
       in go s' iss (o : acc)

-- | Vector SSM as a 'System (->)' with harpie state, input 'AffVec', and full
-- state observation.
ssmSystemVec :: System (->) (Array Double) (Mono AffVec (Array Double))
ssmSystemVec = System $ \(h, d) ->
  let AffVec a b = monoDir d
      h' = zipWith (+) (zipWith (*) a h) b
   in (h', (h', ()))
