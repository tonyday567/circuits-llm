{-# LANGUAGE OverloadedStrings #-}
module Main where

import Circuit.LLM.GPT
import Numeric.LinearAlgebra (Matrix, Vector, konst, sumElements, fromLists, matrix, subMatrix, toRows, fromRows)
import qualified Numeric.LinearAlgebra as LA

main :: IO ()
main = do
  let cfg = GptConfig
        { gptVocabSize = 1000
        , gptNEmbd     = 32
        , gptNHead     = 4
        , gptNLayer    = 2
        }
      seqLen = 8
      rnd rows cols seed =
        fromLists
          [ [ sin (fromIntegral (i * j) * seed :: Double) * 0.02
            | j <- [0 .. cols - 1] ]
          | i <- [0 .. rows - 1]
          ]
      vec n seed = konst 0 n  -- placeholder zeros, unused for now
      mkBlock s = TransformerBlock
        { tbAttnWq = rnd 32 32 (s + 0.1)
        , tbAttnWk = rnd 32 32 (s + 0.2)
        , tbAttnWv = rnd 32 32 (s + 0.3)
        , tbAttnWo = rnd 32 32 (s + 0.4)
        , tbAttnLnGamma = konst 1 32
        , tbAttnLnBeta  = konst 0 32
        , tbFfn  = FeedForward
            { ffW1 = rnd 32 128 (s + 0.7)
            , ffB1 = konst 0 128
            , ffW2 = rnd 128 32 (s + 0.9)
            , ffB2 = konst 0 32
            }
        , tbFfnLnGamma = konst 1 32
        , tbFfnLnBeta  = konst 0 32
        }
      m = Gpt
        { gptWte   = rnd 1000 32 0.01
        , gptWpe   = rnd 1024 32 0.02
        , gptBlocks = [mkBlock 1.0, mkBlock 2.0]
        , gptLnGamma = konst 1 32
        , gptLnBeta  = konst 0 32
        , gptHead   = rnd 32 1000 4.0
        , gptHeadB  = konst 0 1000
        }
      inputIds = [1 .. seqLen]

  putStrLn "Running GPT forward pass..."
  let logits = forward cfg m inputIds
      s = sumElements logits
  putStrLn $ "  sum(logits) = " ++ show s
  putStrLn "OK"
