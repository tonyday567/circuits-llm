-- | Token sampling and auto-regressive text generation.
module Circuit.LLM.Inference
  ( -- * Sampling
    greedySample,
    temperatureSample,
    topKSample,

    -- * Generation
    generate,

    -- * Utilities
    argmax,
    softmaxV,
    lastRow,
  )
where

import Circuit.LLM.BPE (BPEEncoding (..), BPEModel (..), decodeBPE, encodeBPE)
import Circuit.LLM.GPT (Gpt, GptConfig, forward)
import Data.List (sortOn)
import Data.Ord (Down (..))
import Data.Text (Text)
import Data.Text qualified as T
import Data.Vector.Unboxed qualified as VU
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    fromList,
    maxElement,
    maxIndex,
    sumElements,
    toList,
    toRows,
  )
import System.Random (RandomGen, randomR)

----------------------------------------------------------------------
-- Sampling strategies
----------------------------------------------------------------------

-- | Greedy: pick the token with the highest logit.
greedySample :: Vector Double -> Int
greedySample = maxIndex

-- | Temperature sampling.
temperatureSample :: (RandomGen g) => Double -> Vector Double -> g -> (Int, g)
temperatureSample temp logits g =
  let scaled = cmap (/ temp) logits
      probs = softmaxV scaled
      (r, g') = randomR (0, 1) g
   in (sampleCategorical (toList probs) r, g')

-- | Top-K sampling.
topKSample :: (RandomGen g) => Int -> Double -> Vector Double -> g -> (Int, g)
topKSample k temp logits g =
  let vals = toList logits
      indexed = zip [0 :: Int ..] vals
      topk = take k (sortOn (Down . snd) indexed)
      topIds = map fst topk
      topVals = map snd topk
      scaled = map (/ temp) topVals
      probs = softmaxV (fromList scaled)
      (r, g') = randomR (0, 1) g
   in (topIds !! sampleCategorical (toList probs) r, g')

----------------------------------------------------------------------
-- Auto-regressive generation
----------------------------------------------------------------------

-- | Auto-regressive text generation using greedy sampling.
generate ::
  GptConfig -> Gpt -> BPEModel -> Text -> Int -> IO Text
generate cfg model bpe prompt maxNewTokens = do
  let enc = encodeBPE bpe prompt
      tokens = VU.toList (encodedTokens enc)
      initialIds = map fromIntegral tokens -- Word32 -> Int
  resultIds <- go initialIds maxNewTokens
  let resultEnc = enc {encodedTokens = VU.fromList (map fromIntegral resultIds)}
  pure $ decodeBPE bpe (encodedTokens resultEnc)
  where
    go toks 0 = pure toks
    go toks n = do
      let logits = forward cfg model toks
          nextLogits = lastRow logits
          nextToken = greedySample nextLogits
      go (toks ++ [nextToken]) (n - 1)

----------------------------------------------------------------------
-- Utilities
----------------------------------------------------------------------

-- | Softmax over a vector (numerically stable).
softmaxV :: Vector Double -> Vector Double
softmaxV v =
  let mx = maxElement v
      shifted = cmap (\x -> exp (x - mx)) v
      s = sumElements shifted
   in cmap (/ s) shifted

-- | Argmax of a vector.
argmax :: Vector Double -> Int
argmax = maxIndex

-- | Extract the last row of a matrix.
lastRow :: Matrix Double -> Vector Double
lastRow = last . toRows

-- | Sample from a categorical distribution.
sampleCategorical :: [Double] -> Double -> Int
sampleCategorical probs r = go 0 r probs
  where
    go _ _ [] = 0
    go i acc (p : ps)
      | acc < p = i
      | otherwise = go (i + 1) (acc - p) ps
