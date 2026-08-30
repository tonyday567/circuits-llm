module Main where

import Circuit.Body (Body (..))
import Circuit.LLM.Attention (causalMask, multiHeadAttention)
import Circuit.LLM.SSM
  ( Aff (..),
    AffVec (..),
    affComp,
    affCompVec,
    assocSSM,
    assocSSMVec,
    assocScan,
    assocScanVec,
    chunkedScan,
    coupledMultiHeadSSMSystem,
    mooreMorphism,
    multiHeadSSMSystem,
    runMultiHeadSSMSystem,
    runSharedInputMultiHeadSSMSystem,
    seqSSM,
    seqSSMVec,
    ssmProcess,
    ssmSystem,
    ssmSystemVec,
  )
import Circuit.Moore (Moore (..), monoIn)
import Circuit.Process (mooreAsProcess, scan)
import Data.List (foldl', scanl')
import Data.Vector.Unboxed qualified as V
import Harpie.Array (Array, array, mult, shape, zipWith, (!))
import Prelude hiding (zipWith)
import Prelude qualified

approx :: Double -> Double -> Bool
approx x y = abs (x - y) < 1e-9

approxArray :: Array Double -> Array Double -> Bool
approxArray x y =
  and (zipWith (\a b -> abs (a - b) < 1e-9) x y)

approxAff :: Aff -> Aff -> Bool
approxAff (Aff a1 b1) (Aff a2 b2) = approx a1 a2 && approx b1 b2

check :: String -> Bool -> IO Bool
check name ok = do
  putStrLn $ (if ok then "PASS " else "FAIL ") ++ name
  pure ok

-- | Hand-rolled EWMA for exact oracle comparison.
ewmaHand :: Double -> [Double] -> [Double]
ewmaHand alpha xs = case scanl' (\h x -> alpha * x + (1 - alpha) * h) 0 xs of
  (_ : outs) -> outs
  [] -> []

-- ---------------------------------------------------------------------------
-- Constant-A LTI oracles
-- ---------------------------------------------------------------------------

-- | Add the @b@ components of two affine steps with the same @a@.
addB :: Aff -> Aff -> Aff
addB (Aff a1 b1) (Aff a2 b2)
  | approx a1 a2 = Aff a1 (b1 + b2)
  | otherwise = error "addB: a components differ"

-- | Scale the @b@ component of an affine step.
scaleB :: Double -> Aff -> Aff
scaleB k (Aff a b) = Aff a (k * b)

-- | The zero-input affine step for constant decay @a@.
zeroAff :: Double -> Aff
zeroAff a = Aff a 0

-- | Convolution kernel for constant-@a@ SSM: @[1, a, a^2, ...]@.
kernel :: Double -> Int -> [Double]
kernel a n = take n (iterate (* a) 1)

-- | Convolution of two sequences (finite support, zero-padded).
convolve :: [Double] -> [Double] -> [Double]
convolve xs ys =
  let n = min (Prelude.length xs) (Prelude.length ys)
   in [ sum (Prelude.zipWith (*) (Prelude.take (i + 1) xs) (reverse (Prelude.take (i + 1) ys)))
      | i <- [0 .. n - 1]
      ]

-- ---------------------------------------------------------------------------
-- Centrality helpers
-- ---------------------------------------------------------------------------

-- | Premonoidal left-first product of two knot bodies over a shared state.
bodyParL :: ((s, a) -> (s, b)) -> ((s, c) -> (s, d)) -> ((s, (a, c)) -> (s, (b, d)))
bodyParL f g (s, (a, c)) =
  let (s', b) = f (s, a)
      (s'', d) = g (s', c)
   in (s'', (b, d))

-- | Premonoidal right-first product of two knot bodies over a shared state.
bodyParR :: ((s, a) -> (s, b)) -> ((s, c) -> (s, d)) -> ((s, (a, c)) -> (s, (b, d)))
bodyParR f g (s, (a, c)) =
  let (s', d) = g (s, c)
      (s'', b) = f (s', a)
   in (s'', (b, d))

-- | Two bodies are central at the chosen input if left-first and right-first
-- threading agree.
bodyCentral :: (Eq s, Eq b, Eq d) => ((s, a) -> (s, b)) -> ((s, c) -> (s, d)) -> (s, (a, c)) -> Bool
bodyCentral f g input = bodyParL f g input == bodyParR f g input

-- | Independent head-1 body: updates only @h1@, leaves @h2@ unchanged.
independentHead1 :: ((Array Double, Array Double), AffVec) -> ((Array Double, Array Double), Array Double)
independentHead1 ((h1, h2), aff1) =
  let AffVec a1 b1 = aff1
      h1' = zipWith (+) (zipWith (*) a1 h1) b1
   in ((h1', h2), h1')

-- | Independent head-2 body: updates only @h2@, leaves @h1@ unchanged.
independentHead2 :: ((Array Double, Array Double), AffVec) -> ((Array Double, Array Double), Array Double)
independentHead2 ((h1, h2), aff2) =
  let AffVec a2 b2 = aff2
      h2' = zipWith (+) (zipWith (*) a2 h2) b2
   in ((h1, h2'), h2')

-- | Coupled head-1 body: head 1 reads head 2's state.
coupledHead1 :: ((Array Double, Array Double), AffVec) -> ((Array Double, Array Double), Array Double)
coupledHead1 ((h1, h2), aff1) =
  let AffVec a1 b1 = aff1
      cross = zipWith (*) h2 (zipWith (\_ _ -> 0.1) h2 h2)
      h1' = zipWith (+) (zipWith (+) (zipWith (*) a1 h1) b1) cross
   in ((h1', h2), h1')

-- | Coupled head-2 body: independent update.
coupledHead2 :: ((Array Double, Array Double), AffVec) -> ((Array Double, Array Double), Array Double)
coupledHead2 ((h1, h2), aff2) =
  let AffVec a2 b2 = aff2
      h2' = zipWith (+) (zipWith (*) a2 h2) b2
   in ((h1, h2'), h2')

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
      embed =
        array [seqLen, nEmbd] $
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            ++ [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            ++ [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            ++ [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      -- Identity-ish weight matrices for attention.
      wEye =
        array [nEmbd, nEmbd] $
          [ if i == j then 1.0 else 0.0
          | i <- [0 .. nEmbd - 1],
            j <- [0 .. nEmbd - 1]
          ]
      mask4 = causalMask seqLen
      -- SSM layer: fixed diagonal A, B = input embedding row.
      ssmA = array [nEmbd] (replicate nEmbd 0.5)
      ssmH0 = array [nEmbd] (replicate nEmbd 0.0)
      -- Convert each row of embed to an AffVec.
      toRows :: Array Double -> [Array Double]
      toRows x =
        let [n, d] = V.toList (shape x)
         in [array [d] [x ! [r, c] | c <- [0 .. d - 1]] | r <- [0 .. n - 1]]
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
        check "SSM affine action law (constant A)" $
          let seqResult = seqSSM 0 ewmaSteps
              assocResult = assocSSM 0 ewmaSteps
           in and [approx x y | (x, y) <- zip seqResult assocResult],
        check "SSM affine action law (varying A)" $
          let seqResult = seqSSM 0 steps
              assocResult = assocSSM 0 steps
           in and [approx x y | (x, y) <- zip seqResult assocResult],
        check "SSM reorder falsifier: adjacent steps do not commute" $
          let leftFirst = affComp (Aff 0.5 1) (Aff 0.5 2)
              rightFirst = affComp (Aff 0.5 2) (Aff 0.5 1)
           in affA leftFirst == affA rightFirst
                && not (approx (affB leftFirst) (affB rightFirst)),
        check "SSM tree reduction equals prefix scan for all chunk sizes" $
          let nonCommSteps = [Aff 0.5 1, Aff 0.6 2, Aff 0.4 3, Aff 0.7 4, Aff 0.3 5]
              expected = assocScan nonCommSteps
              chunkSizes = [1 .. length nonCommSteps + 1]
           in and
                [ all (uncurry approxAff) (zip (chunkedScan k nonCommSteps) expected)
                | k <- chunkSizes
                ],
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
        check "SSM Moore (,) view typechecks" $
          let (outs, _sF) = mooreMorphism ssmSystemVec h0v vsteps
           in length outs == length vsteps,
        check "SSM Moore (,) scan equals sequential scan" $
          let (sysResult, _sF) = mooreMorphism ssmSystemVec h0v vsteps
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
                && length ssmOut == length affVecs,
        check "SSM multi-head Tensor with per-head inputs/states" $
          let h1 = array [3] [0, 1, 2]
              h2 = array [3] [3, 2, 1]
              head1Steps =
                [ AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [1, 2, 3]),
                  AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [4, 5, 6])
                ]
              head2Steps =
                [ AffVec (array [3] [0.6, 0.6, 0.6]) (array [3] [7, 8, 9]),
                  AffVec (array [3] [0.6, 0.6, 0.6]) (array [3] [10, 11, 12])
                ]
              (tensorOuts, _) = runMultiHeadSSMSystem (h1, h2) (zip head1Steps head2Steps)
              head1Outs = seqSSMVec h1 head1Steps
              head2Outs = seqSSMVec h2 head2Steps
           in length tensorOuts == length head1Steps
                && and
                  [ approxArray o1 expected1 && approxArray o2 expected2
                  | ((o1, o2), expected1, expected2) <- zip3 tensorOuts head1Outs head2Outs
                  ],
        check "SSM shared-input multi-head is diagonal of independent runner" $
          let h0 = array [3] [0, 0, 0]
              sharedSteps =
                [ AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [1, 2, 3]),
                  AffVec (array [3] [0.5, 0.5, 0.5]) (array [3] [4, 5, 6])
                ]
              (tensorOuts, _) = runSharedInputMultiHeadSSMSystem (h0, h0) sharedSteps
              (indOuts, _) = runMultiHeadSSMSystem (h0, h0) [(aff, aff) | aff <- sharedSteps]
           in tensorOuts == indOuts,
        -- -----------------------------------------------------------------------
        -- Moore (,) pointing repair
        -- -----------------------------------------------------------------------
        check "SSM Moore (,) carries h0 as a point" $
          let h0 = 3.0
              affs = [Aff 0.5 1, Aff 0.5 2, Aff 0.5 3]
              procResult = scan (mooreAsProcess ssmSystem h0) affs
              seqResult = seqSSM h0 affs
           in and [approx x y | (x, y) <- zip procResult seqResult],
        -- -----------------------------------------------------------------------
        -- Constant-A LTI oracles
        -- -----------------------------------------------------------------------
        check "SSM superposition on constant A" $
          let a = 0.7
              u = [Aff a 1, Aff a 2, Aff a 3]
              v = [Aff a 4, Aff a 5, Aff a 6]
              left = seqSSM 0 (Prelude.zipWith addB u v)
              right = Prelude.zipWith (+) (seqSSM 0 u) (seqSSM 0 v)
           in and [approx x y | (x, y) <- zip left right],
        check "SSM homogeneity on constant A" $
          let a = 0.7
              affs = [Aff a 1, Aff a 2, Aff a 3]
              k = 2.5
              left = seqSSM 0 (map (scaleB k) affs)
              right = map (* k) (seqSSM 0 affs)
           in and [approx x y | (x, y) <- zip left right],
        check "SSM time invariance on constant A" $
          let a = 0.7
              affs = [Aff a 1, Aff a 2, Aff a 3]
              h0 = 2.0
              k = 2
              zeroPrefix = replicate k (zeroAff a)
              padded = zeroPrefix ++ affs
              shifted = seqSSM h0 padded
              expectedPrefix = map (\i -> a ^ i * h0) [1 .. k]
              rest = seqSSM (a ^ k * h0) affs
           in and [approx x y | (x, y) <- Prelude.zip (Prelude.take k shifted) expectedPrefix]
                && and [approx x y | (x, y) <- Prelude.zip (Prelude.drop k shifted) rest],
        check "SSM recurrence equals convolution for constant A" $
          let a = 0.7
              bs = [1, 2, 3, 4]
              affs = map (Aff a) bs
              seqResult = seqSSM 0 affs
              convResult = convolve (kernel a (length bs)) bs
           in and [approx x y | (x, y) <- zip seqResult convResult],
        -- -----------------------------------------------------------------------
        -- Multi-head centrality pair
        -- -----------------------------------------------------------------------
        check "SSM independent heads are central" $
          let s0 = (array [1] [1], array [1] [2])
              aff1 = AffVec (array [1] [0.5]) (array [1] [1])
              aff2 = AffVec (array [1] [0.6]) (array [1] [2])
           in bodyCentral independentHead1 independentHead2 (s0, (aff1, aff2)),
        check "SSM coupled heads are not central" $
          let s0 = (array [1] [1], array [1] [2])
              aff1 = AffVec (array [1] [0.5]) (array [1] [1])
              aff2 = AffVec (array [1] [0.6]) (array [1] [2])
           in not (bodyCentral coupledHead1 coupledHead2 (s0, (aff1, aff2))),
        check "SSM coupled multi-head Moore (,) typechecks and differs from independent" $
          let h1 = array [1] [1]
              h2 = array [1] [2]
              aff1 = AffVec (array [1] [0.5]) (array [1] [1])
              aff2 = AffVec (array [1] [0.6]) (array [1] [2])
              (indOuts, _) = runMultiHeadSSMSystem (h1, h2) [(aff1, aff2)]
              (coupOuts, _) =
                let Moore (Body f) = coupledMultiHeadSSMSystem
                    go s [] acc = (reverse acc, s)
                    go (x, y) ((a1, a2) : rest) acc =
                      let ((x', y'), ((o1, ()), (o2, ()))) = f ((x, y), (monoIn a1, monoIn a2))
                       in go (x', y') rest ((o1, o2) : acc)
                 in go (h1, h2) [(aff1, aff2)] []
           in indOuts /= coupOuts
      ]
  if and results
    then putStrLn "\nAll tests passed."
    else error "Some tests failed."
