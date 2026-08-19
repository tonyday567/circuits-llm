-- | Multi-head self-attention with causal masking.
module Circuit.LLM.Attention
  ( -- * Core attention
    scaledDotProductAttention,
    multiHeadAttention,

    -- * Components
    softmax,
    causalMask,

    -- * Dimensions
    splitHeads,
    mergeHeads,
  )
where

import Data.Foldable (maximum, sum)
import Data.List (foldl1')
import Data.Vector.Unboxed qualified as V
import Harpie.Array
  ( Array,
    array,
    concatenate,
    drop,
    expand,
    mult,
    reduces,
    reshape,
    shape,
    take,
    transpose,
    zipWith,
  )
import Harpie.Array qualified as H (imap)
import NumHask.Algebra.Additive (Additive)
import NumHask.Algebra.Multiplicative (Multiplicative)
import Prelude hiding (drop, maximum, sum, take, zipWith)

-- | Numerically stable softmax along dimension 0 (row-wise).
softmax :: (Floating a, Ord a, Additive a, Multiplicative a) => Array a -> Array a
softmax x =
  let rowMaxes = reduces [0] maximum x
      expandedMax = expand const rowMaxes (array [nCols] unitVals)
      xShifted = H.imap (\_ v -> exp v) (zipWith (-) x expandedMax)
      rowSums = reduces [0] sum xShifted
      expandedSums = expand const rowSums (array [nCols] unitVals)
   in zipWith (/) xShifted expandedSums
  where
    shapeList = V.toList (shape x)
    nCols =
      case shapeList of
        [_, n] -> n
        _ ->
          -- Input is not two-dimensional; not a reachable state for valid use.
          error "unreachable: softmax expects a 2-D array"
    unitVals = replicate nCols ()

-- | Scaled dot-product attention.
scaledDotProductAttention ::
  (Floating a, Ord a, Additive a, Multiplicative a) =>
  Array a -> Array a -> Array a -> Array a -> Array a
scaledDotProductAttention q k v mask =
  let dk = fromIntegral (last (V.toList (shape k))) :: Double
      scores = H.imap (\_ s -> s / realToFrac (sqrt dk)) (mult q (transpose k))
      masked = zipWith (+) scores mask
      attn = softmax masked
   in mult attn v

-- | Causal mask.
causalMask :: Int -> Array Double
causalMask n =
  array
    [n, n]
    [ if j <= i then 0 else -(1.0 / 0.0)
    | i <- [0 .. n - 1],
      j <- [0 .. n - 1]
    ]

splitHeads :: Int -> Array a -> Array a
splitHeads nHead x =
  let shapeList = V.toList (shape x)
      (seqLen, nEmbd) =
        case shapeList of
          [s, e] -> (s, e)
          _ ->
            -- Input is not two-dimensional; not a reachable state for valid use.
            error "unreachable: splitHeads expects a 2-D array"
      headDim = nEmbd `div` nHead
   in reshape [nHead, seqLen, headDim] (reshape [seqLen, nHead, headDim] x)

mergeHeads :: Array a -> Array a
mergeHeads x =
  let shapeList = V.toList (shape x)
      (nHead, seqLen, headDim) =
        case shapeList of
          [h, s, d] -> (h, s, d)
          _ ->
            -- Input is not three-dimensional; not a reachable state for valid use.
            error "unreachable: mergeHeads expects a 3-D array"
   in reshape [seqLen, nHead * headDim] (reshape [seqLen, nHead, headDim] x)

-- | Multi-head self-attention. Projects x directly with per-head weight slices
-- to avoid deep backpermute chains.
multiHeadAttention ::
  (Floating a, Ord a, Additive a, Multiplicative a) =>
  Int -> Array a -> Array a -> Array a -> Array a -> Array a -> Array a -> Array a
multiHeadAttention nHead x wQ wK wV wO mask =
  let shapeList = V.toList (shape x)
      (seqLen, nEmbd) =
        case shapeList of
          [s, e] -> (s, e)
          _ ->
            -- Input is not two-dimensional; not a reachable state for valid use.
            error "unreachable: multiHeadAttention expects a 2-D array"
      headDim = nEmbd `div` nHead

      -- Project per head: mult x with column slice [n_embd, head_dim] of weights
      headQ h = mult x (take 1 headDim (drop 1 (h * headDim) wQ))
      headK h = mult x (take 1 headDim (drop 1 (h * headDim) wK))
      headV h = mult x (take 1 headDim (drop 1 (h * headDim) wV))

      -- Per-head attention
      attnHeads =
        [ reshape [1, seqLen, headDim] $
            scaledDotProductAttention (headQ h) (headK h) (headV h) mask
        | h <- [0 .. nHead - 1]
        ]

      -- Stack: [n_head, seq_len, head_dim]
      attnH = foldl1' (concatenate 0) attnHeads
   in -- Merge: [seq_len, n_embd] and output projection
      mult (mergeHeads attnH) wO
