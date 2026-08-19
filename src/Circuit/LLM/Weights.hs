{-# LANGUAGE OverloadedStrings #-}

-- | Load GPT-2 weights from a directory of raw float32 binary files.
--
-- Weight files follow a naming convention. Each .f32 file contains
-- raw little-endian IEEE 754 float32 values with no header.
--
-- Required files for a GPT-2 model:
--
-- @
--   wte.f32             [vocab_size, n_embd]
--   wpe.f32             [max_seq_len, n_embd]
--   hN.ln1.gamma.f32    [n_embd]             per block
--   hN.ln1.beta.f32     [n_embd]
--   hN.attn.qkv.w.f32   [n_embd, 3*n_embd]   fused Q,K,V projection
--   hN.attn.qkv.b.f32   [3*n_embd]
--   hN.attn.proj.w.f32  [n_embd, n_embd]
--   hN.attn.proj.b.f32  [n_embd]
--   hN.ln2.gamma.f32    [n_embd]
--   hN.ln2.beta.f32     [n_embd]
--   hN.mlp.fc.w.f32     [n_embd, 4*n_embd]
--   hN.mlp.fc.b.f32     [4*n_embd]
--   hN.mlp.proj.w.f32   [4*n_embd, n_embd]
--   hN.mlp.proj.b.f32   [n_embd]
--   lnf.gamma.f32       [n_embd]
--   lnf.beta.f32        [n_embd]
-- @
module Circuit.LLM.Weights
  ( -- * Loading
    loadGpt2,
    loadGpt2With,
    loadMatrix,
    loadVector,

    -- * GPT-2 architecture dimensions
    Gpt2Size (..),
    sizeConfig,
  )
where

import Circuit.LLM.GPT
  ( FeedForward (..),
    Gpt (..),
    GptConfig (..),
    TransformerBlock (..),
  )
import Data.ByteString (ByteString)
import Data.ByteString qualified as BS
import Data.Word (Word32)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, poke)
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    fromList,
    reshape,
    subMatrix,
    tr,
  )
import System.IO.Unsafe (unsafePerformIO)

----------------------------------------------------------------------
-- GPT-2 sizes
----------------------------------------------------------------------

-- | GPT-2 model size presets.
data Gpt2Size = Gpt2Small | Gpt2Medium | Gpt2Large | Gpt2Xl
  deriving (Show, Eq)

-- | Model configuration for each preset.
sizeConfig :: Gpt2Size -> GptConfig
sizeConfig Gpt2Small = GptConfig 50257 768 12 12
sizeConfig Gpt2Medium = GptConfig 50257 1024 16 24
sizeConfig Gpt2Large = GptConfig 50257 1280 20 36
sizeConfig Gpt2Xl = GptConfig 50257 1600 25 48

----------------------------------------------------------------------
-- Top-level loader
----------------------------------------------------------------------

-- | Load a GPT-2 model with a custom configuration (for non-standard dimensions).
loadGpt2With :: FilePath -> GptConfig -> IO Gpt
loadGpt2With dir cfg = do
  let nEmb = gptNEmbd cfg
      nHead = gptNHead cfg
      nLayer = gptNLayer cfg
      vocab = gptVocabSize cfg
      maxSeq = 1024

  wte <- loadMatrix (dir ++ "/wte.f32") vocab nEmb
  wpe <- loadMatrix (dir ++ "/wpe.f32") maxSeq nEmb
  blocks <- mapM (loadBlock dir nEmb nHead) [0 .. nLayer - 1]
  lnG <- loadVector (dir ++ "/lnf.gamma.f32") nEmb
  lnB <- loadVector (dir ++ "/lnf.beta.f32") nEmb

  pure
    Gpt
      { gptWte = wte,
        gptWpe = wpe,
        gptBlocks = blocks,
        gptLnGamma = lnG,
        gptLnBeta = lnB,
        -- Tie output head to transposed token embeddings (weight tying)
        gptHead = tr wte, -- [n_embd, vocab]
        gptHeadB = zeroVector vocab
      }

-- | Load a full GPT-2 model from a weight directory.
loadGpt2 :: FilePath -> Gpt2Size -> IO Gpt
loadGpt2 dir sz = loadGpt2With dir (sizeConfig sz)

-- | Load one transformer block's weights.
loadBlock :: FilePath -> Int -> Int -> Int -> IO TransformerBlock
loadBlock dir nEmb _nHead h = do
  let pfx = dir ++ "/h" ++ show h ++ "."
      ffMul = 4

  ln1G <- loadVector (pfx ++ "ln1.gamma.f32") nEmb
  ln1B <- loadVector (pfx ++ "ln1.beta.f32") nEmb
  qkvW <- loadMatrix (pfx ++ "attn.qkv.w.f32") nEmb (3 * nEmb)
  _qkvB <- loadVector (pfx ++ "attn.qkv.b.f32") (3 * nEmb)
  projW <- loadMatrix (pfx ++ "attn.proj.w.f32") nEmb nEmb
  _projB <- loadVector (pfx ++ "attn.proj.b.f32") nEmb
  ln2G <- loadVector (pfx ++ "ln2.gamma.f32") nEmb
  ln2B <- loadVector (pfx ++ "ln2.beta.f32") nEmb
  fcW <- loadMatrix (pfx ++ "mlp.fc.w.f32") nEmb (ffMul * nEmb)
  fcB <- loadVector (pfx ++ "mlp.fc.b.f32") (ffMul * nEmb)
  proj2W <- loadMatrix (pfx ++ "mlp.proj.w.f32") (ffMul * nEmb) nEmb
  proj2B <- loadVector (pfx ++ "mlp.proj.b.f32") nEmb

  -- Split fused Q,K,V projection: columns [0:nEmb], [nEmb:2*nEmb], [2*nEmb:3*nEmb]
  let wQ = subMatrix (0, 0) (nEmb, nEmb) qkvW
      wK = subMatrix (0, nEmb) (nEmb, nEmb) qkvW
      wV = subMatrix (0, 2 * nEmb) (nEmb, nEmb) qkvW

  pure
    TransformerBlock
      { tbAttnWq = wQ,
        tbAttnWk = wK,
        tbAttnWv = wV,
        tbAttnWo = projW,
        tbAttnLnGamma = ln1G,
        tbAttnLnBeta = ln1B,
        tbFfn =
          FeedForward
            { ffW1 = fcW,
              ffB1 = fcB,
              ffW2 = proj2W,
              ffB2 = proj2B
            },
        tbFfnLnGamma = ln2G,
        tbFfnLnBeta = ln2B
      }

----------------------------------------------------------------------
-- Binary file I/O
----------------------------------------------------------------------

-- | Load a matrix from a raw float32 (little-endian) file.
loadMatrix :: FilePath -> Int -> Int -> IO (Matrix Double)
loadMatrix path rows_ cols_ = do
  bs <- BS.readFile path
  let n = rows_ * cols_
      expected = n * 4
      floats = take n (parseFloats bs)
  if BS.length bs /= expected
    then error $ path ++ ": expected " ++ show expected ++ " bytes, got " ++ show (BS.length bs)
    else pure $ reshape cols_ $ fromList floats

-- | Load a vector from a raw float32 (little-endian) file.
loadVector :: FilePath -> Int -> IO (Vector Double)
loadVector path n = do
  bs <- BS.readFile path
  let expected = n * 4
      floats = take n (parseFloats bs)
  if BS.length bs /= expected
    then error $ path ++ ": expected " ++ show expected ++ " bytes, got " ++ show (BS.length bs)
    else pure $ fromList floats

-- | Lazy parse of a ByteString into Double values (from float32 LE).
parseFloats :: ByteString -> [Double]
parseFloats bs
  | BS.length bs < 4 = []
  | otherwise =
      let (chunk, rest) = BS.splitAt 4 bs
       in realToFrac (word32ToFloat (readWord32LE chunk)) : parseFloats rest

-- | Read a little-endian 32-bit word from 4 bytes.
readWord32LE :: ByteString -> Word32
readWord32LE bs =
  fromIntegral (BS.index bs 0)
    + fromIntegral (BS.index bs 1) * 0x100
    + fromIntegral (BS.index bs 2) * 0x10000
    + fromIntegral (BS.index bs 3) * 0x1000000

-- | Reinterpret a Word32 as Float via pointer cast.
word32ToFloat :: Word32 -> Float
word32ToFloat w = unsafePerformIO $
  alloca $ \(p :: Ptr Word32) -> do
    poke p w
    peek (castPtr p :: Ptr Float)

-- | Zero vector (placeholder).
zeroVector :: Int -> Vector Double
zeroVector n = fromList (replicate n 0)
