{-# LANGUAGE OverloadedStrings #-}

module Circuit.LLM.BPE
  ( -- * Data Types
    BPEModel (..),
    BPEEncoding (..),
    BPEError (..),

    -- * Model Loading
    loadBPEModel,
    loadBPEModelWithPerf,

    -- * Encoding & Decoding
    encodeBPE,
    encodeBPEWithPerf,
    decodeBPE,
    decodeBPEWithPerf,

    -- * Display Functions
    prettifyBPEModel,
    prettifyEncoding,
  )
where

import Control.Exception (Exception, throwIO)
import Control.Monad (when)
import Data.ByteString (ByteString)
import Data.ByteString qualified as BS
import Data.Char (isAsciiLower, isAsciiUpper, isDigit, isSpace)
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.List (minimumBy)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ord (comparing)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding qualified as TE
import Data.Text.Encoding.Error qualified as TEE
import Data.Text.IO qualified as TIO
import Data.Vector.Unboxed (Vector)
import Data.Vector.Unboxed qualified as V
import Data.Word (Word32)

type Nanos = Integer

-- | BPE model loaded from .model file
-- Stores merge rules, vocabulary, and special token mappings for encoding/decoding
data BPEModel = BPEModel
  { -- | Version string (e.g., "simple-bpe v1")
    bpeVersion :: !Text,
    -- | Regex pattern for text splitting
    bpeRegex :: !ByteString,
    -- | Special tokens → IDs
    bpeSpecialTokens :: !(Map Text Word32),
    -- | Special token IDs → tokens
    bpeReverseSpecial :: !(IntMap Text),
    -- | Pair → (new token, priority)
    bpeMergeRules :: !(Map (Word32, Word32) (Word32, Int)),
    -- | Token ID → bytes (for decoding)
    bpeVocab :: !(IntMap ByteString),
    -- | Highest token ID in vocabulary
    bpeMaxTokenId :: !Word32
  }
  deriving (Eq, Show)

-- | Encoding result with metadata
data BPEEncoding = BPEEncoding
  { -- | Encoded token IDs
    encodedTokens :: !(Vector Word32),
    -- | Original input text
    originalText :: !Text,
    -- | Number of regex chunks processed
    numChunks :: !Int
  }
  deriving (Eq, Show)

-- | BPE operation errors
data BPEError
  = -- | File path and error message
    ModelParseError !FilePath !String
  | -- | Invalid token ID during decode
    InvalidTokenId !Word32
  | -- | Regex pattern compilation error
    RegexCompileError !String
  deriving (Show, Eq)

instance Exception BPEError

-- | Load BPE model from .model file (Rust format)
--
-- File format:
-- Line 1: Version string ("simple-bpe v1")
-- Line 2: Regex pattern for text splitting
-- Line 3: Number of special tokens (integer)
-- Next N lines: Special token and its ID (e.g., "<|endoftext|> 256")
-- Remaining lines: Merge pairs - two token IDs per line (e.g., "65 66")
loadBPEModel :: FilePath -> IO BPEModel
loadBPEModel fp = do
  content <- TIO.readFile fp
  parseModelFile fp content

-- | Load BPE model with performance measurement
--
-- TODO: implement proper perf measurement with pure functions
loadBPEModelWithPerf :: FilePath -> IO (BPEModel, Map Text [Nanos])
loadBPEModelWithPerf fp = do
  model <- loadBPEModel fp
  let timings = Map.empty
  pure (model, timings)

-- | Parse model file content into BPEModel structure
parseModelFile :: FilePath -> Text -> IO BPEModel
parseModelFile fp content = do
  let lns = Text.lines content

  -- Validate minimum number of lines (version + regex + special token count)
  when (length lns < 3) $
    throwIO $
      ModelParseError fp "File too short: missing header"

  case lns of
    (version : regexLine : numSpecialLine : _) -> do
      let regexPattern = TE.encodeUtf8 regexLine
          numSpecialStr = numSpecialLine

      -- Parse number of special tokens
      numSpecial <- case reads (Text.unpack numSpecialStr) of
        [(n, "")] -> pure n
        _ -> throwIO $ ModelParseError fp ("Invalid special token count: " ++ Text.unpack numSpecialStr)

      -- Parse special tokens
      let specialLines = take numSpecial (drop 3 lns)
      specialTokens <- parseSpecialTokens fp specialLines
      let reverseSpecial = IntMap.fromList [(fromIntegral tid, tok) | (tok, tid) <- Map.toList specialTokens]

      -- Parse merge rules
      let mergeLines = drop (3 + numSpecial) lns
      (mergeRules, maxMergeId) <- parseMergeRules fp mergeLines numSpecial

      -- Build vocabulary: base bytes (0-255) + special tokens + merge results
      let vocab = buildVocab specialTokens mergeRules maxMergeId

      pure $
        BPEModel
          { bpeVersion = version,
            bpeRegex = regexPattern,
            bpeSpecialTokens = specialTokens,
            bpeReverseSpecial = reverseSpecial,
            bpeMergeRules = mergeRules,
            bpeVocab = vocab,
            bpeMaxTokenId = maxMergeId
          }
    _ -> throwIO $ ModelParseError fp "File too short: missing header"

-- | Parse special tokens from lines like "<|endoftext|> 256"
parseSpecialTokens :: FilePath -> [Text] -> IO (Map Text Word32)
parseSpecialTokens fp lns = do
  let parseToken line = case Text.words line of
        [tok, idStr] -> case reads (Text.unpack idStr) of
          [(tid, "")] -> pure (tok, tid)
          _ -> throwIO $ ModelParseError fp ("Invalid token ID: " ++ Text.unpack line)
        _ -> throwIO $ ModelParseError fp ("Invalid special token line: " ++ Text.unpack line)
  tokens <- mapM parseToken lns
  pure $ Map.fromList tokens

-- | Parse merge rules from lines like "65 66"
-- Returns (merge rules map, max token ID)
parseMergeRules :: FilePath -> [Text] -> Int -> IO (Map (Word32, Word32) (Word32, Int), Word32)
parseMergeRules fp lns specialCount = do
  let parseRule (idx, line) = case Text.words line of
        [id1Str, id2Str] -> case (reads (Text.unpack id1Str), reads (Text.unpack id2Str)) of
          ([(id1, "")], [(id2, "")]) ->
            let newTokenId = fromIntegral (256 + specialCount + idx)
             in pure ((id1, id2), (newTokenId, idx))
          _ -> throwIO $ ModelParseError fp ("Invalid merge rule: " ++ Text.unpack line)
        _ -> throwIO $ ModelParseError fp ("Invalid merge rule format: " ++ Text.unpack line)

  rules <- mapM parseRule (zip [0 ..] lns)
  let mergeMap = Map.fromList rules
      maxId =
        if null rules
          then fromIntegral (255 + specialCount)
          else maximum [tid | (_, (tid, _)) <- rules]
  pure (mergeMap, maxId)

-- | Build vocabulary from base bytes, special tokens, and merge rules
-- Uses lazy approach: only base bytes (0-255) + special tokens initially
-- Merged token bytes computed on-demand during decode
buildVocab :: Map Text Word32 -> Map (Word32, Word32) (Word32, Int) -> Word32 -> IntMap ByteString
buildVocab specialTokens _mergeRules _maxId =
  let -- Base vocabulary: bytes 0-255
      baseVocab = IntMap.fromList [(i, BS.singleton (fromIntegral i)) | i <- [0 .. 255]]

      -- Special tokens
      specialVocab =
        IntMap.fromList
          [(fromIntegral tid, TE.encodeUtf8 tok) | (tok, tid) <- Map.toList specialTokens]
   in -- Combined: base bytes + special tokens
      -- Merged tokens will be computed lazily during decode
      IntMap.union specialVocab baseVocab

-- | Encode text using BPE model
--
-- Algorithm:
-- 1. Split text by regex pattern into chunks
-- 2. For each chunk:
--    - Check if it's a special token → encode directly
--    - Otherwise: convert to bytes → apply BPE merges
-- 3. Concatenate all results
encodeBPE :: BPEModel -> Text -> BPEEncoding
encodeBPE model text =
  let chunks = splitByRegex (bpeRegex model) text
      encodedChunks = map (encodeChunk model) chunks
      allTokens = V.concat encodedChunks
   in BPEEncoding
        { encodedTokens = allTokens,
          originalText = text,
          numChunks = length chunks
        }

-- | Encode with performance measurement
--
-- TODO: implement proper perf measurement with pure functions
encodeBPEWithPerf :: BPEModel -> Text -> IO (BPEEncoding, Map Text [Nanos])
encodeBPEWithPerf model text = do
  let result = encodeBPE model text
      timings = Map.empty
  pure (result, timings)

-- | Split text by regex pattern
--
-- Fallback implementation: splits on word boundaries and punctuation
splitByRegex :: ByteString -> Text -> [Text]
splitByRegex _pattern text =
  let chunks = Text.words text
      splitChunks = concatMap splitOnPunctuation chunks
   in filter (not . Text.null) splitChunks

-- | Split text on punctuation boundaries
--
-- Example: "hello,world" → ["hello", ",", "world"]
splitOnPunctuation :: Text -> [Text]
splitOnPunctuation t
  | Text.null t = []
  | otherwise =
      let (word, rest) = Text.span (not . isPunct) t
          (punct, remainder) = Text.span isPunct rest
       in [word | not (Text.null word)]
            ++ [punct | not (Text.null punct)]
            ++ splitOnPunctuation remainder
  where
    isPunct c = not (isSpace c) && not (isAlphaNum c)
    isAlphaNum c = isAsciiLower c || isAsciiUpper c || isDigit c

-- | Encode a single text chunk
encodeChunk :: BPEModel -> Text -> Vector Word32
encodeChunk model chunk =
  case Map.lookup chunk (bpeSpecialTokens model) of
    Just tokenId -> V.singleton tokenId
    Nothing -> encodeBytes model (TE.encodeUtf8 chunk)

-- | Encode bytes using BPE merge rules
encodeBytes :: BPEModel -> ByteString -> Vector Word32
encodeBytes model bs =
  let initialTokens = V.fromList [fromIntegral (BS.index bs i) | i <- [0 .. BS.length bs - 1]]
      finalTokens = applyMerges (bpeMergeRules model) initialTokens
   in finalTokens

-- | Apply BPE merges iteratively until no more mergeable pairs
applyMerges :: Map (Word32, Word32) (Word32, Int) -> Vector Word32 -> Vector Word32
applyMerges rules tokens =
  case findBestPair rules tokens of
    Nothing -> tokens
    Just (pair, newToken, _priority) ->
      let merged = mergePair tokens pair newToken
       in applyMerges rules merged

-- | Find the best pair to merge (lowest priority = earliest in training)
findBestPair :: Map (Word32, Word32) (Word32, Int) -> Vector Word32 -> Maybe ((Word32, Word32), Word32, Int)
findBestPair rules tokens =
  let len = V.length tokens
      pairs = [(V.unsafeIndex tokens i, V.unsafeIndex tokens (i + 1)) | i <- [0 .. len - 2]]
      validPairs =
        [ (pair, newToken, priority)
        | pair <- pairs,
          Just (newToken, priority) <- [Map.lookup pair rules]
        ]
   in if null validPairs
        then Nothing
        else Just $ minimumBy (comparing (\(_, _, p) -> p)) validPairs

-- | Merge all occurrences of a pair into a single token
mergePair :: Vector Word32 -> (Word32, Word32) -> Word32 -> Vector Word32
mergePair tokens (tok1, tok2) newToken =
  let len = V.length tokens
      go i acc
        | i >= len = reverse acc
        | i == len - 1 = reverse (V.unsafeIndex tokens i : acc)
        | otherwise =
            let current = V.unsafeIndex tokens i
                next = V.unsafeIndex tokens (i + 1)
             in if current == tok1 && next == tok2
                  then go (i + 2) (newToken : acc)
                  else go (i + 1) (current : acc)
      merged = go 0 []
   in V.fromList merged

-- | Decode token IDs back to text
--
-- Algorithm:
-- 1. For each token ID, lookup bytes in vocabulary
-- 2. If not found in initial vocab, compute recursively from merge rules
-- 3. Concatenate all bytes
-- 4. Decode as UTF-8 (lossy)
decodeBPE :: BPEModel -> Vector Word32 -> Text
decodeBPE model tokens =
  let bytesList = [lookupTokenBytes model tid | tid <- V.toList tokens]
      allBytes = BS.concat bytesList
   in TE.decodeUtf8With TEE.lenientDecode allBytes

-- | Decode with performance measurement
--
-- TODO: implement proper perf measurement with pure functions
decodeBPEWithPerf :: BPEModel -> Vector Word32 -> IO (Text, Map Text [Nanos])
decodeBPEWithPerf model tokens = do
  let result = decodeBPE model tokens
      timings = Map.empty -- Placeholder
  pure (result, timings)

-- | Lookup bytes for a token ID (with lazy vocabulary building)
-- First tries direct lookup in vocab, then computes recursively if needed
lookupTokenBytes :: BPEModel -> Word32 -> ByteString
lookupTokenBytes model tokenId =
  case IntMap.lookup (fromIntegral tokenId) (bpeVocab model) of
    Just bytes -> bytes
    Nothing ->
      -- Token not in vocab - need to compute it from merge rules
      -- This happens for merged tokens not pre-computed
      computeTokenBytes model tokenId

-- | Compute bytes for a merged token ID (recursive)
--
-- Finds the merge rule that created this token and concatenates component bytes
computeTokenBytes :: BPEModel -> Word32 -> ByteString
computeTokenBytes model tokenId =
  case findMergeForToken (bpeMergeRules model) tokenId of
    Nothing -> BS.empty
    Just (tok1, tok2) ->
      let bytes1 = lookupTokenBytes model tok1
          bytes2 = lookupTokenBytes model tok2
       in BS.append bytes1 bytes2

-- | Find the merge rule that produced a given token ID
--
-- Searches through merge rules to find which pair merged into this token
findMergeForToken :: Map (Word32, Word32) (Word32, Int) -> Word32 -> Maybe (Word32, Word32)
findMergeForToken rules targetId =
  let matches = [(pair, tid) | (pair, (tid, _)) <- Map.toList rules, tid == targetId]
   in case matches of
        ((pair, _) : _) -> Just pair
        [] -> Nothing

-- | Pretty-print BPE model information
prettifyBPEModel :: BPEModel -> String
prettifyBPEModel model =
  unlines $
    [ "BPE Model:",
      "  Version: " ++ Text.unpack (bpeVersion model),
      "  Regex: " ++ show (bpeRegex model),
      "  Special tokens: " ++ show (Map.size (bpeSpecialTokens model)),
      "  Merge rules: " ++ show (Map.size (bpeMergeRules model)),
      "  Vocab size: " ++ show (IntMap.size (bpeVocab model)),
      "  Max token ID: " ++ show (bpeMaxTokenId model),
      "",
      "Special tokens:"
    ]
      ++ specialTokenList
  where
    specialTokenList =
      [ "  " ++ Text.unpack tok ++ " -> " ++ show tid
      | (tok, tid) <- Map.toList (bpeSpecialTokens model)
      ]

-- | Pretty-print encoding result
prettifyEncoding :: BPEEncoding -> String
prettifyEncoding enc =
  unlines
    [ "BPE Encoding:",
      "  Original text: " ++ show (originalText enc),
      "  Token count: " ++ show (V.length (encodedTokens enc)),
      "  Chunks processed: " ++ show (numChunks enc),
      "",
      "Token IDs:",
      "  " ++ show (V.toList (encodedTokens enc))
    ]
