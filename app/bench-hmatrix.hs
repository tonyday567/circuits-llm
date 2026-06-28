module Main where

import Numeric.LinearAlgebra
  ( Matrix, Vector, (<>), (|||), cmap, fromLists, fromRows, scale, toRows, tr
  , konst, cols, rows, sumElements
  )
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Data (maxElement)
import Control.Exception (evaluate)
import System.CPUTime (getCPUTime)
import Text.Printf (printf)
import Prelude hiding ((<>), (|||))

main :: IO ()
main = do
  -- Matmul
  let ma = konst 1 (64, 16) :: Matrix Double
      mb = konst 1 (16, 64) :: Matrix Double
  bench "matmul [64,16]x[16,64]" $ sumElements (ma <> mb)

  -- Softmax
  let smIn = fromLists [[fromIntegral (i*j)/100 | j<-[1..64]] | i<-[1..64]] :: Matrix Double
  bench "softmax [64,64]" $ sumElements (softmaxH smIn)

  -- SDPA
  let q = fromLists [[fromIntegral (i*j)/100 | j<-[1..16]] | i<-[1..64]] :: Matrix Double
      k = q
      v = fromLists [[fromIntegral (i+j) | j<-[1..16]] | i<-[1..64]] :: Matrix Double
      m = fromLists [[if j<=i then 0 else -1/0 | j<-[1..64]] | i<-[1..64]] :: Matrix Double
  bench "sdpa [64,16]" $ sumElements (sdpaH q k v m)

  -- Full MHA at GPT-2 scale
  let seqL = 64; nEmb = 64; nHd = 4; hDim = 16
      x = konst 1 (seqL, nEmb) :: Matrix Double
      wq = konst 1 (nEmb, nEmb); wk = konst 1 (nEmb, nEmb)
      wv = konst 1 (nEmb, nEmb); wo = konst 1 (nEmb, nEmb)
      mask = fromLists [[if j<=i then 0 else -1/0 | j<-[1..seqL]] | i<-[1..seqL]]
  bench "mha [64,64] 4h" $ sumElements (mhaH nHd x wq wk wv wo mask)

  putStrLn "\nDONE"

bench :: String -> Double -> IO ()
bench name val = do
  s <- evaluate val  -- force, get value
  _ <- evaluate val  -- warmup (force again)
  t0 <- getCPUTime
  _ <- evaluate val
  t1 <- getCPUTime
  printf "  %-25s %8.3f ms  (sum=%s)\n" name
    (fromIntegral (t1-t0) / 1e6 :: Double) (show s)

-- | Row-wise softmax using hmatrix (row-by-row via toRows/fromRows).
softmaxH :: Matrix Double -> Matrix Double
softmaxH x =
  fromRows [softmaxVec row | row <- toRows x]
  where
    softmaxVec v =
      let mx = LA.maxElement v
          shifted = cmap (\xi -> exp (xi - mx)) v
          s = LA.sumElements shifted
      in  scale (1 / s) shifted

-- | Scaled dot-product attention.
sdpaH :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
sdpaH q k v mask =
  let dk = fromIntegral (cols k) :: Double
      scores = scale (1 / sqrt dk) (q <> tr k)
      masked = scores + mask  -- element-wise
      attn = softmaxH masked
  in  attn <> v

-- | Multi-head attention. Projects per-head using column slices of weights.
mhaH :: Int -> Matrix Double -> Matrix Double -> Matrix Double
     -> Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
mhaH nHead x wQ wK wV wO mask =
  let -- Per-head attention
      heads =
        [ sdpaH (x <> wSlice wQ h) (x <> wSlice wK h) (x <> wSlice wV h) mask
        | h <- [0 .. nHead-1]
        ]
      concat = fromColumns heads
  in  concat <> wO
  where
    headDim = cols wQ `div` nHead
    wSlice w h =
      let rows_ = rows w
      in  fromLists [[w `LA.atIndex` (r, c) | c <- [h*headDim .. (h+1)*headDim - 1]] | r <- [0 .. rows_ - 1]]
    fromColumns [a] = a
    fromColumns (a:as) = a ||| fromColumns as
    fromColumns [] = error "no heads"
