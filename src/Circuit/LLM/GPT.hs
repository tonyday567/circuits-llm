{-# LANGUAGE OverloadedStrings #-}

-- | GPT-2 architecture using hmatrix for linear algebra.
module Circuit.LLM.GPT
  ( -- * Normalisation and activation
    layerNorm,
    gelu,

    -- * Transformer block
    FeedForward (..),
    TransformerBlock (..),
    transformerBlock,

    -- * Full model
    GptConfig (..),
    Gpt (..),
    forward,
  )
where

import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromLists,
    fromRows,
    rows,
    scale,
    subMatrix,
    sumElements,
    toList,
    toRows,
    tr,
    (|||),
  )
import Numeric.LinearAlgebra qualified as LA
import Numeric.LinearAlgebra.Data (maxElement)
import Prelude hiding (drop, sum, take)

-- | Layer normalisation along the last axis.
layerNorm :: Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
layerNorm x gamma beta eps = fromRows $ map normRow (toRows x)
  where
    g = toList gamma
    b = toList beta
    d = fromIntegral (cols x)
    normRow xi =
      let mu = sumElements xi / d
          xi' = cmap (subtract mu) xi
          var = sumElements (xi' * xi') / d + eps
          invStd = 1 / sqrt var
          scaled = scale invStd xi'
       in LA.fromList $ zipWith3 (\v gi bi -> v * gi + bi) (toList scaled) g b

-- | GELU activation (approximation).
gelu :: Matrix Double -> Matrix Double
gelu = cmap f
  where
    f v =
      let x' = 1.59577 * v * (1 + 0.044715 * v * v)
       in v / (1 + exp (-x'))

-- | Feed-forward network.
data FeedForward = FeedForward
  { ffW1 :: Matrix Double,
    ffB1 :: Vector Double,
    ffW2 :: Matrix Double,
    ffB2 :: Vector Double
  }

feedForward :: FeedForward -> Matrix Double -> Matrix Double
feedForward ff x =
  let h = gelu (x LA.<> ffW1 ff + LA.asRow (ffB1 ff))
   in h LA.<> ffW2 ff + LA.asRow (ffB2 ff)

-- | Causal mask.
causalMask :: Int -> Matrix Double
causalMask n = fromLists [[if j <= i then 0 else -(1 / 0) | j <- [0 .. n - 1]] | i <- [0 .. n - 1]]

-- | Row-wise softmax.
softmax :: Matrix Double -> Matrix Double
softmax x = fromRows [softmaxRow row | row <- toRows x]
  where
    softmaxRow v =
      let mx = maxElement v
          shifted = cmap (\xi -> exp (xi - mx)) v
          s = sumElements shifted
       in scale (1 / s) shifted

-- | Scaled dot-product attention.
scaledDotProductAttention :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
scaledDotProductAttention q k v mask =
  let dk = fromIntegral (cols k) :: Double
      scores = scale (1 / sqrt dk) (q LA.<> tr k)
      masked = scores + mask
      attn = softmax masked
   in attn LA.<> v

-- | Multi-head self-attention.
multiHeadAttention :: Int -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
multiHeadAttention nHead x wQ wK wV wO mask =
  let headDim = cols wQ `div` nHead
      wSlice w h = subMatrix (0, h * headDim) (rows w, headDim) w
      heads =
        [ scaledDotProductAttention (x LA.<> wSlice wQ h) (x LA.<> wSlice wK h) (x LA.<> wSlice wV h) mask
        | h <- [0 .. nHead - 1]
        ]
   in concatCols heads LA.<> wO
  where
    concatCols [a] = a
    concatCols (a : as) = a ||| concatCols as
    concatCols [] = error "no heads"

-- | One transformer block.
data TransformerBlock = TransformerBlock
  { tbAttnWq :: Matrix Double,
    tbAttnWk :: Matrix Double,
    tbAttnWv :: Matrix Double,
    tbAttnWo :: Matrix Double,
    tbAttnLnGamma :: Vector Double,
    tbAttnLnBeta :: Vector Double,
    tbFfn :: FeedForward,
    tbFfnLnGamma :: Vector Double,
    tbFfnLnBeta :: Vector Double
  }

transformerBlock :: Int -> TransformerBlock -> Matrix Double -> Matrix Double
transformerBlock nHead tb x =
  let attnOut =
        multiHeadAttention
          nHead
          x
          (tbAttnWq tb)
          (tbAttnWk tb)
          (tbAttnWv tb)
          (tbAttnWo tb)
          (causalMask (rows x))
      attnRes = x + attnOut
      attnNorm = layerNorm attnRes (tbAttnLnGamma tb) (tbAttnLnBeta tb) 1e-5
      ffnOut = feedForward (tbFfn tb) attnNorm
      ffnRes = attnNorm + ffnOut
   in layerNorm ffnRes (tbFfnLnGamma tb) (tbFfnLnBeta tb) 1e-5

-- | GPT-2 model configuration.
data GptConfig = GptConfig
  { gptVocabSize :: Int,
    gptNEmbd :: Int,
    gptNHead :: Int,
    gptNLayer :: Int
  }

-- | GPT-2 model parameters.
data Gpt = Gpt
  { gptWte :: Matrix Double,
    gptWpe :: Matrix Double,
    gptBlocks :: [TransformerBlock],
    gptLnGamma :: Vector Double,
    gptLnBeta :: Vector Double,
    gptHead :: Matrix Double,
    gptHeadB :: Vector Double
  }

-- | Full forward pass.
forward :: GptConfig -> Gpt -> [Int] -> Matrix Double
forward cfg m inputIds =
  let seqLen = length inputIds
      tokEmb = fromRows [toRows (gptWte m) !! i | i <- inputIds]
      posEmb = subMatrix (0, 0) (seqLen, cols (gptWpe m)) (gptWpe m)
      x = tokEmb + posEmb
      x' = foldl (flip (transformerBlock (gptNHead cfg))) x (gptBlocks m)
      x'' = layerNorm x' (gptLnGamma m) (gptLnBeta m) 1e-5
   in x'' LA.<> gptHead m + LA.asRow (gptHeadB m)
