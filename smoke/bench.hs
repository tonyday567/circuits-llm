{-# LANGUAGE OverloadedStrings #-}

-- | Benchmark BPE encoding vs Python.
module Main where

import Circuit.LLM.BPE
import Circuit.Meter.Time (ticksION)
import Control.Monad (replicateM_)
import Data.ByteString qualified as BS
import Data.Map.Strict qualified as Map
import Data.Text qualified as T
import Data.Text.Encoding qualified as TE
import Data.Text.IO qualified as TIO
import Data.Vector.Unboxed qualified as V
import System.IO (hFlush, hPutStrLn, stderr, stdout)
import Text.Printf (printf)

main :: IO ()
main = do
  hPutStrLn stderr "loading model ..."
  model <- loadBPEModel modelPath

  hPutStrLn stderr "reading test data ..."
  text <- TIO.readFile textPath
  let textLen = T.length text

  hPutStrLn stderr "benchmarking ...\n"

  putStrLn $ "Model: " ++ modelPath
  putStrLn $ "  Merge rules: " ++ show (Map.size (bpeMergeRules model))
  putStrLn $ "  Special tokens: " ++ show (Map.size (bpeSpecialTokens model))
  putStrLn $ "  Max token ID: " ++ show (bpeMaxTokenId model)

  putStrLn $ "\nTest data: " ++ textPath
  putStrLn $ "  Characters: " ++ show textLen
  putStrLn $ "  Bytes (UTF-8): " ++ show (BS.length (TE.encodeUtf8 text))

  -- Single encode timing
  putStrLn "\n=== single encode ==="
  (coldNs, enc) <- ticksION 1 (pure $! encodeBPE model text)
  let coldMs = fromIntegral coldNs / 1e6 :: Double
      numTokens = V.length (encodedTokens enc)
  printf
    "  %7.2f ms   %d tokens   %7.0f tok/s\n"
    coldMs
    numTokens
    (fromIntegral numTokens / (coldMs / 1000))

  -- Repeated encode
  let nWarm = 5
      nMeas = 20
  hPutStrLn stderr $ "warming up (" ++ show nWarm ++ " encodes) ..."
  replicateM_ nWarm (pure $! encodeBPE model text)
  putStrLn $ "\n=== steady state (" ++ show nMeas ++ " encodes) ==="
  (avgNs, _) <- ticksION nMeas (pure $! encodeBPE model text)
  let avgMs = fromIntegral avgNs / 1e6 :: Double
  printf
    "  %7.2f ms avg   %d tokens   %7.0f tok/s   %7.0f chars/s\n"
    avgMs
    numTokens
    (fromIntegral numTokens / (avgMs / 1000))
    (fromIntegral textLen / (avgMs / 1000))

  -- Decode
  putStrLn "\n=== decode ==="
  (decodeNs, _) <- ticksION nMeas (pure $! decodeBPE model (encodedTokens enc))
  printf "  %7.2f ms avg\n" (fromIntegral decodeNs / 1e6 :: Double)

  -- Roundtrip
  let decoded = decodeBPE model (encodedTokens enc)
  putStrLn "\n=== roundtrip ==="
  putStrLn $ "  " ++ (if text == decoded then "PASS" else "FAIL  (len " ++ show (T.length text) ++ " vs " ++ show (T.length decoded) ++ ")")

  hFlush stdout

modelPath :: FilePath
modelPath = "/Users/tonyday567/other/building-from-scratch/bpe/data/tok276.model"

textPath :: FilePath
textPath = "/Users/tonyday567/other/building-from-scratch/bpe/data/text.txt"
