{-# LANGUAGE OverloadedStrings #-}
import Circuit.LLM.BPE
import qualified Data.Text as T
import qualified Data.Text.IO as TIO

main = do
  m <- loadBPEModel "/Users/tonyday567/other/building-from-scratch/bpe/data/tok276.model"
  TIO.putStrLn $ T.pack (prettifyBPEModel m)
  let e = encodeBPE m "hello world, how are you?"
  TIO.putStrLn $ T.pack (prettifyEncoding e)
  let d = decodeBPE m (encodedTokens e)
  TIO.putStrLn $ "decoded: " <> d
  -- Test with elab text
  let elabText = "intent \10230 know what is in the tent"
      e2 = encodeBPE m elabText
  TIO.putStrLn $ "\nelab test: " <> elabText
  TIO.putStrLn $ "tokens: " <> T.pack (show (encodedTokens e2))
  TIO.putStrLn $ "decoded: " <> decodeBPE m (encodedTokens e2)
