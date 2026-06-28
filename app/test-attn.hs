module Main where

import Circuit.LLM.Attention
import Harpie.Array (Array, array, index, shape)
import Text.Printf (printf)

(!) :: Array a -> [Int] -> a
(!) = Harpie.Array.index

main :: IO ()
main = do
  -- === softmax ===
  let x = array [2, 3] [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] :: Array Double
  let sm = softmax x
  printf "softmax [2,3]: row0 sum=%.4f row1 sum=%.4f %s\n"
    (sum [sm ! [0, c] | c <- [0 .. 2]])
    (sum [sm ! [1, c] | c <- [0 .. 2]])
    (if abs (sum [sm ! [0, c] | c <- [0 .. 2]] - 1.0) < 1e-10 then "PASS" else "FAIL")

  -- === causal mask ===
  let mask = causalMask 3
  printf "mask [0,0]=%.1f mask[0,1]=%.1f %s\n"
    (mask ! [0, 0]) (mask ! [0, 1])
    (if mask ! [0, 0] == 0.0 && mask ! [0, 1] < (-1e300) then "PASS" else "FAIL")

  -- === sdpa ===
  let q = array [2, 4] [1, 0, 0, 0, 0, 1, 0, 0] :: Array Double
      k = q
      v = array [2, 3] [1, 2, 3, 4, 5, 6] :: Array Double
      m = array [2, 2] [0, -1.0 / 0.0, 0, 0] :: Array Double
      out = scaledDotProductAttention q k v m
  printf "sdpa shape: %s out[0]: %s %s\n"
    (show (shape out))
    (show [out ! [0, c] | c <- [0 .. 2]])
    (if abs (out ! [0, 1] - 2.0) < 1e-10 then "PASS" else "FAIL")

  -- === multi-head ===
  let seqLen = 4; nEmbd = 8; nHead = 2
  let wq = array [nEmbd, nEmbd] [1 | _ <- [1 .. nEmbd * nEmbd]] :: Array Double
      wk = wq; wv = wq; wo = wq
      emb = array [seqLen, nEmbd] [fromIntegral i | i <- [1 .. seqLen * nEmbd]] :: Array Double
      cmask = causalMask seqLen
      mha = multiHeadAttention nHead emb wq wk wv wo cmask
  printf "mha shape: %s %s\n"
    (show (shape mha))
    (if shape mha == [seqLen, nEmbd] then "PASS" else "FAIL")

  putStrLn "\nAll tests complete."
