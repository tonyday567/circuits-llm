{-# LANGUAGE OverloadedStrings #-}

-- | Local compositional reverse-mode AD over hmatrix.
--
-- The idea is lifted from circuits-ad's 'Diff' arrow, but specialised to
-- dense 'hmatrix' matrices/vectors so that the heavy linear algebra still
-- calls BLAS.
--
-- Both causal (GPT-style) and bidirectional (BERT-style) transformer bodies
-- are provided.  They share the same block structure and differ only in the
-- attention mask that is added to the raw QK^T scores.
module Circuit.LLM.Diff
  ( -- * Differentiable operations
    DiffP,
    primForward,
    primBackward,
    (@.),
    residual,
    splitP,
    joinP,

    -- * Primitive layers
    linearP,
    geluP,
    softmaxP,
    layerNormP,
    multiHeadAttentionP,

    -- * Transformer block
    BlockParams (..),
    blockParamsFromBlock,
    blockDiffP,
    gptBlockDiffP,
    bertBlockDiffP,

    -- * Full model bodies
    GptParams (..),
    gptParamsFromModel,
    gptDiffP,
    bertDiffP,

    -- * Forward helpers (also used by Backprop)
    geluF,
    softmaxStableF,
    layerNormF,
    concatColsF,
    broadcastBias,
    causalMaskF,
    accumulateCols,
    subMatrixW,
    zeroMatrix,
  )
where

import Circuit.Diff.Param (TensorPrim (..))
import Circuit.Diff.Param qualified as ADP
import Circuit.LLM.GPT (FeedForward (..), Gpt (..), GptConfig (..), TransformerBlock (..))
import Data.List (foldl1')
import Numeric.LinearAlgebra
  ( Matrix,
    Vector,
    cmap,
    cols,
    fromList,
    fromLists,
    fromRows,
    maxElement,
    reshape,
    rows,
    scale,
    sumElements,
    toList,
    toRows,
    tr,
    (|||),
  )
import Numeric.LinearAlgebra qualified as LA

----------------------------------------------------------------------
-- Compositional differentiable arrow
----------------------------------------------------------------------

-- | A differentiable operation with parameters @p@, input @a@ and output @b@.
--
-- Running forward produces the output.  The backward pass, given the output
-- cotangent, produces the input cotangent and parameter gradients.
--
-- This is exactly the same shape as 'Circuit.Diff.Param.TensorPrim' from
-- circuits-ad; we use a local type synonym so the rest of the module keeps
-- its paired-parameter composition style.
type DiffP = TensorPrim

-- | Sequential composition: @f '@.' g@ means "first @g@, then @f@".
infixr 9 @.

(@.) :: DiffP p2 b c -> DiffP p1 a b -> DiffP (p1, p2) a c
f2 @. f1 =
  TensorPrim
    { primForward = \(p1, p2) a -> primForward f2 p2 (primForward f1 p1 a),
      primBackward = \(p1, p2) a dc ->
        let b = primForward f1 p1 a
            (db, dp2) = primBackward f2 p2 b dc
            (da, dp1) = primBackward f1 p1 a db
         in (da, (dp1, dp2))
    }

-- | Add a residual connection around an operation.
--   forward:  y = x + op(x)
--   backward: dx = dy + dOp
residual :: (Num a) => DiffP p a a -> DiffP p a a
residual op =
  TensorPrim
    { primForward = \p a -> a + primForward op p a,
      primBackward = \p a dy ->
        let (daOp, dp) = primBackward op p a dy
         in (dy + daOp, dp)
    }

-- | Split a parameter tuple so the left and right halves can be used
--   by independent parallel branches.
splitP :: DiffP p1 a1 b1 -> DiffP p2 a2 b2 -> DiffP (p1, p2) (a1, a2) (b1, b2)
splitP f1 f2 =
  TensorPrim
    { primForward = \(p1, p2) (a1, a2) -> (primForward f1 p1 a1, primForward f2 p2 a2),
      primBackward = \(p1, p2) (a1, a2) (db1, db2) ->
        let (da1, dp1) = primBackward f1 p1 a1 db1
            (da2, dp2) = primBackward f2 p2 a2 db2
         in ((da1, da2), (dp1, dp2))
    }

-- | Pair two operations that share the same input type but produce
--   independent outputs.  This is useful for attention Q/K/V from one x.
joinP :: (Num a) => DiffP p1 a b1 -> DiffP p2 a b2 -> DiffP (p1, p2) a (b1, b2)
joinP f1 f2 =
  TensorPrim
    { primForward = \(p1, p2) a -> (primForward f1 p1 a, primForward f2 p2 a),
      primBackward = \(p1, p2) a (db1, db2) ->
        let (da1, dp1) = primBackward f1 p1 a db1
            (da2, dp2) = primBackward f2 p2 a db2
         in (da1 + da2, (dp1, dp2))
    }

----------------------------------------------------------------------
-- Primitive layers
----------------------------------------------------------------------

-- | Linear layer: y = xW + b
linearP :: DiffP (Matrix Double, Vector Double) (Matrix Double) (Matrix Double)
linearP =
  TensorPrim
    { primForward = \(w, b) x -> x LA.<> w + broadcastBias b (rows x),
      primBackward = \(w, _) x dy ->
        let dx = dy LA.<> tr w
            dw = tr x LA.<> dy
            db = fromList [sumElements col | col <- LA.toColumns dy]
         in (dx, (dw, db))
    }

-- | GELU activation.  Migrated to use @circuits-ad:DiffP@.
geluP :: ADP.DiffP () (Matrix Double) (Matrix Double)
geluP =
  ADP.fromPrim $
    ADP.TensorPrim
      { ADP.primForward = \_ x -> geluF x,
        ADP.primBackward = \_ x dy -> (geluBwd x dy, ())
      }

-- | Row-wise softmax.
softmaxP :: DiffP () (Matrix Double) (Matrix Double)
softmaxP =
  TensorPrim
    { primForward = \_ x -> softmaxStableF x,
      primBackward = \_ x dy ->
        let probs = softmaxStableF x
         in (softmaxBwd probs dy, ())
    }
  where
    softmaxBwd probs gradOut =
      fromRows $ zipWith softmaxRowBwd (toRows probs) (toRows gradOut)
      where
        softmaxRowBwd p go =
          let pv = toList p; gov = toList go; dot = sum (zipWith (*) pv gov)
           in fromList $ zipWith (\pi_ goi -> pi_ * (goi - dot)) pv gov

-- | Layer normalisation.
layerNormP :: Double -> DiffP (Vector Double, Vector Double) (Matrix Double) (Matrix Double)
layerNormP eps =
  TensorPrim
    { primForward = \(gamma, beta) x -> layerNormF x gamma beta eps,
      primBackward = \(gamma, beta) x dy ->
        let (dx, dgamma, dbeta) = layerNormBwd x gamma beta eps dy
         in (dx, (dgamma, dbeta))
    }

-- | Multi-head self-attention with a supplied attention mask.
--   Parameters are (Wq, Wk, Wv, Wo).  The mask is added to the raw QK^T
--   scores before softmax; use 'causalMaskF' for GPT and a zero matrix for
--   fully bidirectional BERT-style attention.
multiHeadAttentionP ::
  Int ->
  Int ->
  Double ->
  Matrix Double ->
  DiffP
    (Matrix Double, Matrix Double, Matrix Double, Matrix Double)
    (Matrix Double)
    (Matrix Double)
multiHeadAttentionP nHead seqLen _eps mask =
  TensorPrim
    { primForward = \(wq, wk, wv, wo) x ->
        let hDim = cols wq `div` nHead
            heads =
              [ let wQh = subMatrixW wq 0 (h * hDim) (rows wq) hDim
                    wKh = subMatrixW wk 0 (h * hDim) (rows wk) hDim
                    wVh = subMatrixW wv 0 (h * hDim) (rows wv) hDim
                    q = x LA.<> wQh
                    k = x LA.<> wKh
                    v = x LA.<> wVh
                    scores = scale (1 / sqrt (fromIntegral hDim :: Double)) (q LA.<> tr k)
                    probs = softmaxStableF (scores + mask)
                 in probs LA.<> v
              | h <- [0 .. nHead - 1]
              ]
            ctx = concatColsF heads
         in ctx LA.<> wo,
      primBackward = \(wq, wk, wv, wo) x dOut ->
        let hDim = cols wq `div` nHead
            dk = fromIntegral hDim :: Double
            -- Recompute forward intermediates
            an = x
            headFwd h =
              let wQh = subMatrixW wq 0 (h * hDim) (rows wq) hDim
                  wKh = subMatrixW wk 0 (h * hDim) (rows wk) hDim
                  wVh = subMatrixW wv 0 (h * hDim) (rows wv) hDim
                  q = an LA.<> wQh
                  k = an LA.<> wKh
                  v = an LA.<> wVh
                  scores = scale (1 / sqrt dk) (q LA.<> tr k)
                  probs = softmaxStableF (scores + mask)
               in (h, wQh, wKh, wVh, q, k, v, probs)
            headStates = map headFwd [0 .. nHead - 1]
            ctx = concatColsF [probs LA.<> v | (_, _, _, _, _, _, v, probs) <- headStates]
            -- Backward through output projection
            (gradCtx, gWo, _) = linearBwd ctx wo dOut
            -- Per-head backward
            headBwd (h, wQh, wKh, wVh, q, k, v, probs) =
              let goh = subMatrixW gradCtx 0 (h * hDim) seqLen hDim
                  gradProbs = goh LA.<> tr v
                  gradV = tr probs LA.<> goh
                  gradScores = softmaxBwd probs gradProbs
                  gradScoresS = scale (1 / sqrt dk) gradScores
                  gradQ = gradScoresS LA.<> k
                  gradK = tr gradScoresS LA.<> q
                  (gradXq, gWQ, _) = linearBwd an wQh gradQ
                  (gradXk, gWK, _) = linearBwd an wKh gradK
                  (gradXv, gWV, _) = linearBwd an wVh gradV
               in (gradXq + gradXk + gradXv, gWQ, gWK, gWV)
            headGrads = map headBwd headStates
            gradX = foldl1' (+) [gx | (gx, _, _, _) <- headGrads]
            gWqFull = accumulateCols wq [(h * hDim, gwq) | (h, (_, gwq, _, _)) <- zip [0 ..] headGrads]
            gWkFull = accumulateCols wk [(h * hDim, gwk) | (h, (_, _, gwk, _)) <- zip [0 ..] headGrads]
            gWvFull = accumulateCols wv [(h * hDim, gwv) | (h, (_, _, _, gwv)) <- zip [0 ..] headGrads]
         in (gradX, (gWqFull, gWkFull, gWvFull, gWo))
    }
  where
    softmaxBwd probs gradOut =
      fromRows $ zipWith softmaxRowBwd (toRows probs) (toRows gradOut)
      where
        softmaxRowBwd p go =
          let pv = toList p; gov = toList go; dot = sum (zipWith (*) pv gov)
           in fromList $ zipWith (\pi_ goi -> pi_ * (goi - dot)) pv gov
    linearBwd x w gradY =
      let gradX = gradY LA.<> tr w
          gradW = tr x LA.<> gradY
          gradB = fromList [sumElements col | col <- LA.toColumns gradY]
       in (gradX, gradW, gradB)

----------------------------------------------------------------------
-- Transformer block
----------------------------------------------------------------------

-- | Flat parameter bundle for a transformer block.
data BlockParams = BlockParams
  { bpAttnWq, bpAttnWk, bpAttnWv, bpAttnWo :: Matrix Double,
    bpAttnLnGamma, bpAttnLnBeta :: Vector Double,
    bpFfnW1, bpFfnW2 :: Matrix Double,
    bpFfnB1, bpFfnB2 :: Vector Double,
    bpFfnLnGamma, bpFfnLnBeta :: Vector Double
  }

blockParamsFromBlock :: TransformerBlock -> BlockParams
blockParamsFromBlock tb =
  let ff = tbFfn tb
   in BlockParams
        { bpAttnWq = tbAttnWq tb,
          bpAttnWk = tbAttnWk tb,
          bpAttnWv = tbAttnWv tb,
          bpAttnWo = tbAttnWo tb,
          bpAttnLnGamma = tbAttnLnGamma tb,
          bpAttnLnBeta = tbAttnLnBeta tb,
          bpFfnW1 = ffW1 ff,
          bpFfnB1 = ffB1 ff,
          bpFfnW2 = ffW2 ff,
          bpFfnB2 = ffB2 ff,
          bpFfnLnGamma = tbFfnLnGamma tb,
          bpFfnLnBeta = tbFfnLnBeta tb
        }

-- | Post-LN transformer block as a differentiable operation.
--
--   x -> Attention -> + -> LN -> FFN -> + -> LN -> y
--
-- The implementation is written directly against 'BlockParams' rather than
-- composed with '@.' so that the parameter type stays flat and readable.
-- The attention mask is supplied explicitly, so the same block can be used
-- for causal (GPT) or bidirectional (BERT) attention.
blockDiffP ::
  Int ->
  Int ->
  Double ->
  Matrix Double ->
  DiffP BlockParams (Matrix Double) (Matrix Double)
blockDiffP nHead seqLen eps mask =
  TensorPrim
    { primForward = \p x ->
        let attnOut =
              primForward
                (multiHeadAttentionP nHead seqLen eps mask)
                (bpAttnWq p, bpAttnWk p, bpAttnWv p, bpAttnWo p)
                x
            postAttn = x + attnOut
            attnNorm = layerNormF postAttn (bpAttnLnGamma p) (bpAttnLnBeta p) eps
            ffnHidden = primForward linearP (bpFfnW1 p, bpFfnB1 p) attnNorm
            ffnActivated = geluF ffnHidden
            ffnOut = primForward linearP (bpFfnW2 p, bpFfnB2 p) ffnActivated
            ffnRes = attnNorm + ffnOut
         in layerNormF ffnRes (bpFfnLnGamma p) (bpFfnLnBeta p) eps,
      primBackward = \p x dy ->
        -- Forward recompute
        let attnOut =
              primForward
                (multiHeadAttentionP nHead seqLen eps mask)
                (bpAttnWq p, bpAttnWk p, bpAttnWv p, bpAttnWo p)
                x
            postAttn = x + attnOut
            attnNorm = layerNormF postAttn (bpAttnLnGamma p) (bpAttnLnBeta p) eps
            ffnHidden = primForward linearP (bpFfnW1 p, bpFfnB1 p) attnNorm
            ffnActivated = geluF ffnHidden
            ffnOut = primForward linearP (bpFfnW2 p, bpFfnB2 p) ffnActivated
            ffnRes = attnNorm + ffnOut
            -- Backward through final LN
            (dFfnRes, (dgFfnGamma, dgFfnBeta)) =
              primBackward (layerNormP eps) (bpFfnLnGamma p, bpFfnLnBeta p) ffnRes dy
            -- Backward through FFN residual: dFfnRes splits to attnNorm and FFN
            (dAttnNorm2, (dgFfnW2, dgFfnB2)) =
              primBackward linearP (bpFfnW2 p, bpFfnB2 p) ffnActivated dFfnRes
            (dFfnHidden, ()) =
              snd (ADP.runDiffP geluP () ffnHidden) dAttnNorm2
            (dAttnNorm1, (dgFfnW1, dgFfnB1)) =
              primBackward linearP (bpFfnW1 p, bpFfnB1 p) attnNorm dFfnHidden
            dAttnNorm = dFfnRes + dAttnNorm1
            -- Backward through attn LN
            (dPostAttn, (dgAttnGamma, dgAttnBeta)) =
              primBackward (layerNormP eps) (bpAttnLnGamma p, bpAttnLnBeta p) postAttn dAttnNorm
            -- Backward through attn residual: dPostAttn splits to x and attnOut
            (dXfromAttn, (dgWq, dgWk, dgWv, dgWo)) =
              primBackward
                (multiHeadAttentionP nHead seqLen eps mask)
                (bpAttnWq p, bpAttnWk p, bpAttnWv p, bpAttnWo p)
                x
                dPostAttn
            dX = dPostAttn + dXfromAttn
         in ( dX,
              BlockParams
                { bpAttnWq = dgWq,
                  bpAttnWk = dgWk,
                  bpAttnWv = dgWv,
                  bpAttnWo = dgWo,
                  bpAttnLnGamma = dgAttnGamma,
                  bpAttnLnBeta = dgAttnBeta,
                  bpFfnW1 = dgFfnW1,
                  bpFfnB1 = dgFfnB1,
                  bpFfnW2 = dgFfnW2,
                  bpFfnB2 = dgFfnB2,
                  bpFfnLnGamma = dgFfnGamma,
                  bpFfnLnBeta = dgFfnBeta
                }
            )
    }

-- | Causal transformer block (GPT-style).
gptBlockDiffP :: Int -> Int -> Double -> DiffP BlockParams (Matrix Double) (Matrix Double)
gptBlockDiffP nHead seqLen eps = blockDiffP nHead seqLen eps (causalMaskF seqLen)

-- | Bidirectional transformer block (BERT-style).  Attention can attend to
-- all positions in the sequence.
bertBlockDiffP :: Int -> Int -> Double -> DiffP BlockParams (Matrix Double) (Matrix Double)
bertBlockDiffP nHead seqLen eps = blockDiffP nHead seqLen eps (zeroMatrix seqLen seqLen)

----------------------------------------------------------------------
-- Full model bodies
----------------------------------------------------------------------

-- | Flat parameter bundle for the whole transformer body (excluding
--   embeddings, which are handled separately).
data GptParams = GptParams
  { gpBlocks :: [BlockParams],
    gpLnGamma, gpLnBeta :: Vector Double,
    gpHead :: Matrix Double,
    gpHeadB :: Vector Double
  }

gptParamsFromModel :: Gpt -> GptParams
gptParamsFromModel m =
  GptParams
    { gpBlocks = map blockParamsFromBlock (gptBlocks m),
      gpLnGamma = gptLnGamma m,
      gpLnBeta = gptLnBeta m,
      gpHead = gptHead m,
      gpHeadB = gptHeadB m
    }

-- | Generic transformer body.  The supplied block constructor chooses the
--   attention mask (causal for GPT, bidirectional for BERT).
bodyDiffP ::
  GptConfig ->
  Int ->
  Double ->
  (Int -> Int -> Double -> DiffP BlockParams (Matrix Double) (Matrix Double)) ->
  DiffP GptParams (Matrix Double) (Matrix Double)
bodyDiffP cfg seqLen eps blockCtor =
  TensorPrim
    { primForward = \p x ->
        let xBlocks = foldl' applyBlock x (gpBlocks p)
            xFinalNorm = layerNormF xBlocks (gpLnGamma p) (gpLnBeta p) eps
         in xFinalNorm LA.<> gpHead p + broadcastBias (gpHeadB p) seqLen,
      primBackward = \p x dy ->
        let fwdStack = foldl' (\acc bp -> acc ++ [applyBlock (last acc) bp]) [x] (gpBlocks p)
            xBlocks = last fwdStack
            xFinalNorm = layerNormF xBlocks (gpLnGamma p) (gpLnBeta p) eps
            -- Backward through output projection
            (gradXFinalNorm, (gHead, gHeadB)) =
              primBackward linearP (gpHead p, gpHeadB p) xFinalNorm dy
            -- Backward through final layer norm
            (gradXBlocks, (gLnGamma, gLnBeta)) =
              primBackward (layerNormP eps) (gpLnGamma p, gpLnBeta p) xBlocks gradXFinalNorm
            -- Backward through blocks
            xPrevs = case reverse fwdStack of (_ : xs) -> xs; [] -> []
            (gradX0, blockGradsRev) =
              goBwd gradXBlocks (reverse (gpBlocks p)) xPrevs
         in ( gradX0,
              GptParams
                { gpBlocks = reverse blockGradsRev,
                  gpLnGamma = gLnGamma,
                  gpLnBeta = gLnBeta,
                  gpHead = gHead,
                  gpHeadB = gHeadB
                }
            )
    }
  where
    nHead = gptNHead cfg
    blockP = blockCtor nHead seqLen eps
    applyBlock xIn bp = primForward blockP bp xIn
    goBwd gradX [] _ = (gradX, [])
    goBwd gradX (bp : bps') (xPrev : xs) =
      let (dx, dbp) = primBackward blockP bp xPrev gradX
          (dxFinal, dbps) = goBwd dx bps' xs
       in (dxFinal, dbp : dbps)
    goBwd _ _ _ = error "bodyDiffP: mismatched block stack"

-- | Full GPT model as a differentiable operation.
--
-- Input is the already-embedded token matrix @x0@; output is logits.
-- Embeddings are kept outside because the lookup is not differentiable wrt
-- token IDs, only wrt the embedding matrices.
gptDiffP ::
  GptConfig ->
  Int ->
  Double ->
  DiffP GptParams (Matrix Double) (Matrix Double)
gptDiffP cfg seqLen eps = bodyDiffP cfg seqLen eps gptBlockDiffP

-- | Full BERT-style bidirectional model as a differentiable operation.
--   The architecture is identical to GPT except that attention is not
--   causally masked, so every position can attend to every other position.
bertDiffP ::
  GptConfig ->
  Int ->
  Double ->
  DiffP GptParams (Matrix Double) (Matrix Double)
bertDiffP cfg seqLen eps = bodyDiffP cfg seqLen eps bertBlockDiffP

----------------------------------------------------------------------
-- Forward helpers (shared with Backprop)
----------------------------------------------------------------------

geluF :: Matrix Double -> Matrix Double
geluF = cmap f
  where
    f v = let z = 1.59577 * v * (1 + 0.044715 * v * v) in v / (1 + exp (-z))

-- | GELU backward helper (used by the migrated 'geluP').
geluBwd :: Matrix Double -> Matrix Double -> Matrix Double
geluBwd x gradY =
  let xs = toList (LA.flatten x)
      gs = toList (LA.flatten gradY)
      deriv v =
        let a = 1.59577
            b = 0.044715
            z = a * v * (1 + b * v * v)
            phi = 1 / (1 + exp (-z))
            phi' = phi * (1 - phi) * a * (1 + 3 * b * v * v)
         in phi + v * phi'
   in reshape (cols x) (fromList (zipWith (*) gs (map deriv xs)))

softmaxStableF :: Matrix Double -> Matrix Double
softmaxStableF m =
  fromRows
    [ let mx = maxElement row; s = cmap (\x -> exp (x - mx)) row
       in scale (1 / sumElements s) s
    | row <- toRows m
    ]

layerNormF :: Matrix Double -> Vector Double -> Vector Double -> Double -> Matrix Double
layerNormF x gamma beta eps =
  let d = fromIntegral (cols x)
      mu = fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu
      var = fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (flip scale) (toRows xm) (toList invStd)
   in fromRows
        [ fromList $ zipWith3 (\v g b -> v * g + b) (toList row) (toList gamma) (toList beta)
        | row <- toRows xHat
        ]

layerNormBwd ::
  Matrix Double ->
  Vector Double ->
  Vector Double ->
  Double ->
  Matrix Double ->
  (Matrix Double, Vector Double, Vector Double)
layerNormBwd x gamma _beta eps gradY =
  let mu = fromList [sumElements row / d | row <- toRows x]
      xm = x - LA.asColumn mu
      var = fromList [sumElements (row * row) / d | row <- toRows xm]
      invStd = cmap (\v -> 1 / sqrt (v + eps)) var
      xHat = fromRows $ zipWith (flip scale) (toRows xm) (toList invStd)
      gammaList = toList gamma
      gradXHat = fromRows [fromList $ zipWith (*) (toList go) gammaList | go <- toRows gradY]
      gGamma = fromList [sumElements (go * xh) | (go, xh) <- zip (LA.toColumns gradY) (LA.toColumns xHat)]
      gBeta = fromList [sumElements go | go <- LA.toColumns gradY]
      gradX =
        fromRows
          [ gradRow
              (toRows gradXHat !! i)
              (toRows xm !! i)
              (toList invStd !! i)
              (toList var !! i)
          | i <- [0 .. rows x - 1]
          ]
   in (gradX, gGamma, gBeta)
  where
    d = fromIntegral (cols x)
    features = cols x
    gradRow gxh xi_minus_mu is v =
      let gInvStd = sumElements (gxh * xi_minus_mu)
          gVar = gInvStd * (-0.5) * (v + eps) ** (-1.5)
          gXm = scale is gxh + scale (2 * gVar / d) xi_minus_mu
          gMu = -sumElements gXm
       in gXm + fromList (replicate features (gMu / d))

concatColsF :: [Matrix Double] -> Matrix Double
concatColsF [a] = a
concatColsF (a : as) = a ||| concatColsF as
concatColsF [] = error "no heads"

broadcastBias :: Vector Double -> Int -> Matrix Double
broadcastBias b n = reshape (LA.size b) (fromList (concat (replicate n (toList b))))

subMatrixW :: Matrix Double -> Int -> Int -> Int -> Int -> Matrix Double
subMatrixW m r c rows' cols' = LA.subMatrix (r, c) (rows', cols') m

causalMaskF :: Int -> Matrix Double
causalMaskF n = fromLists [[if j <= i then 0 else -(1 / 0) | j <- [0 .. n - 1]] | i <- [0 .. n - 1]]

accumulateCols :: Matrix Double -> [(Int, Matrix Double)] -> Matrix Double
accumulateCols template patches =
  let full = zeroMatrix (rows template) (cols template)
   in foldl'
        ( \acc (offset, patch) ->
            reshape
              (cols acc)
              ( fromList
                  [ if c >= offset && c < offset + cols patch
                      then
                        (toList (LA.flatten acc) !! (r * cols acc + c))
                          + (toList (LA.flatten patch) !! (r * cols patch + (c - offset)))
                      else toList (LA.flatten acc) !! (r * cols acc + c)
                  | r <- [0 .. rows acc - 1],
                    c <- [0 .. cols acc - 1]
                  ]
              )
        )
        full
        patches

zeroMatrix :: Int -> Int -> Matrix Double
zeroMatrix r c = reshape c (fromList (replicate (r * c) 0))
