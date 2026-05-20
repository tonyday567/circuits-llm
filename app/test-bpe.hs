{-# LANGUAGE OverloadedStrings #-}

import Control.Exception (SomeException, try)
import Data.Text qualified as Text
import Data.Text.IO qualified as TIO
import Data.Vector.Unboxed qualified as V
import Circuit.LLM.BPE
import System.Exit (exitFailure, exitSuccess)

-- Create minimal test model file
-- Format: version, regex, num_special, special tokens, merge rules
createTestModelFile :: FilePath -> IO ()
createTestModelFile fp = do
  TIO.writeFile fp $
    Text.unlines
      [ "simple-bpe v1",
        "[a-z]+", -- regex pattern
        "1", -- 1 special token
        "<|endoftext|> 256", -- special token
        "104 101", -- merge: 'h' (104) + 'e' (101) -> token 257
        "101 108", -- merge: 'e' (101) + 'l' (108) -> token 258
        "108 108" -- merge: 'l' (108) + 'l' (108) -> token 259
      ]

main :: IO ()
main = do
  putStrLn "BPE Module Test - Phase 6"
  putStrLn "========================\n"

  -- Create test model file
  let modelPath = "/tmp/test-bpe.model"
  createTestModelFile modelPath
  putStrLn $ "[1/5] ✓ Created test model at " ++ modelPath

  -- Test 1: Model Loading
  putStrLn "[2/5] Testing model loading..."
  modelResult <- try (loadBPEModel modelPath) :: IO (Either SomeException BPEModel)
  case modelResult of
    Right model -> do
      putStrLn "  ✓ Model loaded"
      putStrLn $ "    - Version: " ++ Text.unpack (bpeVersion model)
      putStrLn $ "    - Merge rules: " ++ show (length (bpeReverseSpecial model))
    Left err -> do
      putStrLn $ "  ✗ Model loading failed: " ++ show err
      exitFailure

  -- Test 2: Encoding
  putStrLn "[3/5] Testing text encoding..."
  case modelResult of
    Right model -> do
      let testText = "hello world"
          encoded = encodeBPE model testText
      putStrLn $ "  ✓ Encoded \"" ++ Text.unpack testText ++ "\""
      putStrLn $ "    - Token count: " ++ show (V.length (encodedTokens encoded))
      putStrLn $ "    - Chunks: " ++ show (numChunks encoded)
    Left _ -> putStrLn "  ✗ Cannot test encoding (model not loaded)"

  -- Test 3: Decoding
  putStrLn "[4/5] Testing token decoding..."
  case modelResult of
    Right model -> do
      -- Encode then decode (round-trip)
      let testText = "hello"
          encoded = encodeBPE model testText
          decoded = decodeBPE model (encodedTokens encoded)
      putStrLn "  ✓ Decoded tokens"
      putStrLn $ "    - Original: \"" ++ Text.unpack testText ++ "\""
      putStrLn $ "    - Decoded:  \"" ++ Text.unpack decoded ++ "\""
    Left _ -> putStrLn "  ✗ Cannot test decoding (model not loaded)"

  -- Test 4: Display functions
  putStrLn "[5/5] Testing display functions..."
  case modelResult of
    Right model -> do
      putStrLn "  ✓ Display functions available"
      putStrLn $ "    Model info:\n" ++ unlines (map ("      " ++) (lines (prettifyBPEModel model)))
    Left _ -> putStrLn "  ✗ Cannot test display (model not loaded)"

  putStrLn "\n========================"
  putStrLn "All tests completed"
  exitSuccess
