# Tokenization as Inductive Bias: Project Evolution and Findings

**Authors:** Thomas Wyatt, Mohab Yousef, Tad Sligar
**Course:** COMP6970 — Generative AI, Auburn University
**Date:** March 28, 2026

---

## 1. Project Goal

Compare how three tokenization strategies — Word-level, Byte-Pair Encoding (BPE), and SentencePiece (Unigram LM) — affect a Transformer language model's ability to predict rare and morphologically derived words, when all other architectural components are held constant.

---

## 2. Evolution of the Experiment

### Iteration 1: Original Implementation (v1)

**Configuration:**
- Dataset: WikiText-2 (~2M tokens)
- Vocab size: 18,000
- Model: 4-layer BERT MLM, hidden size 256, intermediate 1024 (~7.9M params)
- Training: 5,000 steps, batch size 16
- Evaluation: Top-1, Top-5, Top-10 accuracy

**Results:** Complete prediction failure — 0.0% accuracy across every metric, every tokenizer, every benchmark. Mean ranks ranged from 871 to 10,607 out of 18K vocab. Models predicted stop words ("the", "of", "and") for every masked position.

**Diagnosis:** Multiple issues identified:
1. **Severely undertrained** — 5,000 steps covered roughly 3-5 epochs (not the "<3% of one epoch" initially estimated, but still insufficient for convergence)
2. **Evaluation bugs** — String vs. ID comparison mismatch in word tokenizer top-1 metric; word tokenizer piece accuracy hardcoded to 1.0; BPE/SP evaluation dropped 96% of rare words due to single-token-only constraint (BPE had only 17 usable examples vs. 500 for word-level)
3. **Missing deliverables** — MorphyNet loaded but never used for evaluation; no embedding similarity analysis; no analysis breakdowns by word length or frequency

### Iteration 2: Bug Fixes + Training Improvement (v1 patched)

**Changes made:**
- Fixed word tokenizer top-1 to use ID comparison (`gold_id in topk_ids[:1]`) instead of string comparison
- Fixed word tokenizer `avg_piece_accuracy` to reflect actual prediction accuracy instead of hardcoded 1.0
- Added unified multi-token evaluation (`evaluate_unified`) so BPE/SP could be fairly evaluated on all words, not just single-token words
- Changed training from 5,000 max steps to 20 full epochs
- Added validation dataset to Trainer with per-epoch evaluation
- Added `load_best_model_at_end=True` for automatic best-model selection
- Added training/validation loss curve plotting
- Added MorphyNet morphological evaluation cell
- Added detailed analysis breakdowns (by word length, subword piece count, frequency)

**Results (18K vocab, 256 hidden, 20 epochs):** Still 0.0% top-k accuracy across the board. However, training loss did converge properly:
- Word: val loss 5.82 (perplexity ~337)
- BPE: val loss 6.30 (perplexity ~545)
- SentencePiece: val loss 5.79 (perplexity ~326)

SentencePiece showed the only nonzero piece accuracy in reconstruction (0.9–4.8%), and achieved the lowest loss overall. But no model could rank any target word in the top 10.

**Key insight:** The models were learning (losses well below random baseline of ln(18000) = 9.8), but a perplexity of ~330 means the model spreads probability across ~330 candidates — far too diffuse for top-10 accuracy.

### Iteration 3: Model Capacity + Evaluation Expansion (v2)

**Diagnosis of remaining issues:**
- 59% of model parameters (4.6M of 7.9M) were in the embedding table — the actual transformer had only 3.1M parameters to learn contextual patterns
- Predicting from 18,000 candidates was too hard for a small model
- Top-10 was too narrow a bar to show differentiation

**Changes made:**
- Reduced vocabulary from 18,000 to 8,000 (cuts embedding params from 4.6M to 2.0M)
- Increased hidden size from 256 to 512 (transformer capacity 3.1M → 12.6M)
- Increased intermediate size from 1024 to 2048 (standard 4x ratio)
- Total model size: ~17.1M params with 24% in embeddings (vs. 59% before)
- Added top-50 and top-100 evaluation metrics
- Kept WikiText-2, 20 epochs

**Results (8K vocab, 512 hidden, 20 epochs):** Breakthrough — the word tokenizer achieved meaningful prediction accuracy for the first time:

| Benchmark | Word Top-10 | BPE Top-10 | SP Top-10 |
|---|---|---|---|
| Frequent words | **25.0%** | 0.0% | 0.0% |
| Medium-frequency | **15.4%** | 0.0% | 0.0% |

Word tokenizer validation loss dropped to 3.07 (perplexity ~22), a massive improvement from 5.82 (perplexity ~337). BPE and SP improved only modestly (6.12 and 5.67 respectively).

**However, a new problem emerged:** With 8K vocabulary, the word tokenizer had zero usable rare word examples on WikiText — all rare words (freq 1–9) fall outside the top 8,000 most frequent words. The evaluation filter required `word in word2id_split`, which eliminated rare words for all tokenizers, not just word-level.

### Iteration 4: Evaluation Completeness (v3) — Final Version

**Changes made:**
- Decoupled rare/medium/frequent word candidate filters from word tokenizer vocabulary — each tokenizer now independently determines whether it can handle a word
- Added embedding similarity evaluation using MorphyNet morphological pairs (cosine similarity between base and derived word embeddings), fulfilling the proposal's "closest neighbors to a morphed word" requirement
- Added random baseline comparison for embedding similarity
- Added affix-type breakdown (prefix vs. suffix)

**Results:** Full evaluation across all benchmarks — see Section 3.

### Iteration 5: Ablation Study

To isolate which architectural change drove the improvement, we ran two additional configurations:

| Config | Vocab | Hidden | Intermediate | Params | Notebook |
|---|---|---|---|---|---|
| A (v1 original) | 18K | 256 | 1024 | 7.9M | `v1 results` |
| B (vocab only) | 8K | 256 | 1024 | 5.3M | `ablation_vocab8k_hidden256` |
| C (hidden only) | 18K | 512 | 2048 | 22.2M | `ablation_vocab18k_hidden512` |
| D (both, v3) | 8K | 512 | 2048 | 17.1M | `v3` |

All four configurations use WikiText-2, 20 epochs, identical 4-layer BERT MLM architecture, and the same evaluation pipeline.

**Results — Word Tokenizer Validation Loss:**

| Config | Val Loss | Perplexity |
|---|---|---|
| A: 18K/256 | 5.82 | ~337 |
| B: 8K/256 | 5.43 | ~228 |
| C: 18K/512 | 3.41 | ~30 |
| D: 8K/512 | 3.09 | ~22 |

**Results — Word Tokenizer Top-k Accuracy (Medium-Frequency Words):**

| Config | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|
| A: 18K/256 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| B: 8K/256 | 0.0% | 0.0% | 0.3% | 2.0% | 3.7% |
| C: 18K/512 | **6.0%** | **12.8%** | **16.3%** | **31.5%** | **41.0%** |
| D: 8K/512 | 3.7% | 11.8% | 15.4% | 32.3% | 39.9% |

**Results — Word Tokenizer Top-k Accuracy (Frequent Words):**

| Config | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|
| A: 18K/256 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| B: 8K/256 | 0.0% | 0.0% | 0.0% | 5.0% | 22.5% |
| C: 18K/512 | **7.5%** | **22.5%** | **32.5%** | **57.5%** | **65.0%** |
| D: 8K/512 | 0.0% | 17.5% | 25.0% | 40.0% | 50.0% |

**Results — WikiText Rare Word Evaluation (Word Tokenizer):**

| Config | Usable Examples | Top-1 | Top-10 | Top-100 |
|---|---|---|---|---|
| A: 18K/256 | 500 | 0.0% | 0.0% | 0.0% |
| B: 8K/256 | 0 | N/A | N/A | N/A |
| C: 18K/512 | **166** | **3.0%** | **7.2%** | **21.7%** |
| D: 8K/512 | 0 | N/A | N/A | N/A |

**Results — SentencePiece Validation Loss:**

| Config | SP Val Loss | SP Perplexity |
|---|---|---|
| A: 18K/256 | 5.79 | ~326 |
| B: 8K/256 | 5.67 | ~291 |
| C: 18K/512 | **5.05** | **~156** |
| D: 8K/512 | 5.67 | ~291 |

**Results — SP Reconstruction Piece Accuracy (WikiText Rare Words):**

| Config | SP Piece Acc |
|---|---|
| A: 18K/256 | 0.9% |
| B: 8K/256 | 1.2% |
| C: 18K/512 | **7.2%** |
| D: 8K/512 | 1.2% |

**Results — Embedding Similarity Delta (morphological - random):**

| Config | Word | BPE | SP |
|---|---|---|---|
| B: 8K/256 | +0.033 | +0.529 | +0.479 |
| C: 18K/512 | +0.045 | +0.527 | +0.431 |
| D: 8K/512 | +0.025 | +0.515 | +0.467 |

**Ablation Conclusions:**

1. **Hidden size increase (256→512) is the dominant factor.** Config C outperforms Config B on every metric. Going from 256 to 512 hidden dropped word perplexity from ~337 to ~30, while vocab reduction alone only dropped it from ~337 to ~228.

2. **Vocab reduction alone provides marginal improvement.** Config B achieves only 0.3% top-10 on medium words — barely above Config A's 0.0%. The 256-hidden transformer lacks the capacity to exploit the smaller prediction space.

3. **18K vocab (Config C) actually outperforms 8K vocab (Config D) on prediction.** Config C achieves 32.5% top-10 on frequent words vs. D's 25.0%, and 16.3% top-10 on medium words vs. D's 15.4%. More importantly, Config C can evaluate rare words (166 usable examples, 7.2% top-10) while Config D cannot (0 usable).

4. **18K vocab enables the project's core question to be answered.** Only Configs A and C have usable rare word examples for the word tokenizer. Config C — with its 22.2M parameters and 18K vocab — is the only configuration where all three tokenizers can be meaningfully compared on rare words.

5. **SP benefits more from hidden size than vocab reduction.** SP val loss drops from 5.79 to 5.05 with hidden size increase (Config C) but only to 5.67 with vocab reduction (Config B). SP's reconstruction piece accuracy on rare words jumps to 7.2% in Config C vs. 1.2% in Configs B and D.

---

## 3. Final Results

Given the ablation findings, we present results from two configurations: Config C (18K/512) as the **primary configuration** because it enables rare word evaluation across all tokenizers, and Config D (8K/512) as a **complementary configuration** that demonstrates the OOV trade-off.

### 3.1 Training Performance

**Config C (18K vocab, 512 hidden, 22.2M params):**

| Model | Training Loss | Validation Loss | Perplexity |
|---|---|---|---|
| Word | 4.07 | **3.41** | **~30** |
| SentencePiece | 5.59 | **5.05** | **~156** |
| BPE | 6.23 | 6.28 | ~533 |

**Config D (8K vocab, 512 hidden, 17.1M params):**

| Model | Training Loss | Validation Loss | Perplexity |
|---|---|---|---|
| Word | 3.81 | **3.09** | **~22** |
| SentencePiece | 5.65 | 5.67 | ~291 |
| BPE | 6.18 | 6.12 | ~456 |

### 3.1 Training Performance

| Model | Training Loss | Validation Loss | Perplexity | Training Time |
|---|---|---|---|---|
| Word | 3.81 | **3.09** | **~22** | 12 min |
| SentencePiece | 5.65 | 5.67 | ~291 | 19 min |
| BPE | 6.18 | 6.12 | ~456 | 16 min |

All three models share identical architecture (17.1M parameters) and training budget (20 epochs on WikiText-2). Word-level achieves dramatically lower loss because it predicts whole words from a concentrated 8K vocabulary, while BPE and SP must predict subword fragments.

### 3.2 Masked Word Prediction Accuracy

Results shown for Config C (18K/512) as primary. Config D (8K/512) in parentheses where different.

**Frequent words (freq >= 100):**

| Tokenizer | Usable | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|---|
| Word | 40 | **7.5%** (0.0%) | **22.5%** (17.5%) | **32.5%** (25.0%) | **57.5%** (40.0%) | **65.0%** (50.0%) |
| BPE | 39 | 0.0% | 0.0% | 0.0% | 2.6% (7.7%) | 10.3% (15.4%) |
| SP | 29 | 0.0% | 0.0% | 3.4% (0.0%) | 13.8% (6.9%) | 34.5% (27.6%) |

**Medium-frequency words (freq 20–50):**

| Tokenizer | Usable | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|---|
| Word | 400 (356) | **6.0%** (3.7%) | **12.8%** (11.8%) | **16.3%** (15.4%) | **31.5%** (32.3%) | **41.0%** (39.9%) |
| BPE | 390 (24) | 0.0% | 0.0% | 0.0% | 1.8% (4.2%) | 3.6% (4.2%) |
| SP | 151 (140) | 0.7% (0.0%) | 0.7% (0.0%) | 0.7% (0.0%) | 4.0% (2.1%) | 7.9% (3.6%) |

**WikiText rare words (freq 1–9):**

| Tokenizer | Config C Usable | Config C Top-1 | Config C Top-10 | Config C Top-100 | Config D Usable |
|---|---|---|---|---|---|
| Word | **166** | **3.0%** | **7.2%** | **21.7%** | 0 (all UNK) |
| BPE | 500 (11 single) | 0.0% | 0.0% | 0.0% | 500 |
| SP | 485 (82 single) | 0.0% | 0.0% | 0.6% | 450 |

With 8K vocabulary (Config D), the word tokenizer has zero usable rare word examples — all rare words fall outside the top 8,000 most frequent words and map to UNK. With 18K vocabulary (Config C), 166 rare words remain in vocabulary, and the word tokenizer achieves 3.0% top-1 and 7.2% top-10 accuracy on them. This demonstrates both the OOV problem and that word-level tokenization *can* predict rare words when they are in vocabulary.

**Shakespeare rare words (domain-shift):**

| Tokenizer | Usable | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|---|
| Word | 258 (171) | 0.0% | 0.8% (0.6%) | 1.2% (0.6%) | 3.5% (2.9%) | 5.4% (8.8%) |
| BPE | 494 | 0.0% | 0.0% | 0.0% | 0.0% | 1.0% (0.0%) |
| SP | 463 (418) | 0.0% | 0.0% | 0.0% | 0.0% | 0.7% (0.2%) |

**MorphyNet-derived words:**

| Tokenizer | Usable | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|---|
| Word | 444 (231) | **2.9%** (2.2%) | **7.9%** (7.4%) | **10.8%** (11.7%) | **25.5%** (29.4%) | **32.9%** (38.1%) |
| BPE | 500 | 0.0% | 0.0% | 0.0% | 0.6% (0.2%) | 1.9% (0.6%) |
| SP | 493 (464) | 0.0% | 0.8% (0.0%) | 0.8% (0.0%) | 3.9% (0.0%) | 7.8% (0.9%) |

### 3.3 Reconstruction Accuracy (Multi-Token Words)

Config C results (Config D in parentheses):

| Benchmark | Word Exact Match | BPE Piece Acc | SP Piece Acc |
|---|---|---|---|
| WikiText rare | **1.0%** (0.0%) | 0.0% (0.2%) | **7.2%** (1.2%) |
| Shakespeare rare | 0.0% | 0.0% (0.15%) | **4.4%** (4.0%) |
| MorphyNet | **2.6%** (1.0%) | 0.0% (0.15%) | **3.1%** (0.5%) |

SentencePiece consistently achieves the highest piece-level reconstruction accuracy. With 18K vocab and 512 hidden (Config C), SP reaches 7.2% piece accuracy on WikiText rare words — a 6x improvement over Config D's 1.2%. This suggests the larger model captures more of SP's subword structure.

### 3.4 Embedding Similarity (Morphological Relationships)

This evaluation measures whether each model's embeddings capture morphological relationships by computing cosine similarity between base-derived word pairs from MorphyNet (e.g., "sport"→"sporting", "improve"→"improved").

**Config C (18K/512):**

| Metric | Word | BPE | SP |
|---|---|---|---|
| Morphological pair similarity | 0.144 | **0.734** | **0.561** |
| Random pair similarity (baseline) | 0.099 | 0.208 | 0.130 |
| **Delta (morphological signal)** | +0.045 | **+0.527** | **+0.431** |
| Usable pairs | 22 | 2,000 | 2,000 |

**Config D (8K/512):**

| Metric | Word | BPE | SP |
|---|---|---|---|
| Morphological pair similarity | 0.117 | **0.758** | **0.624** |
| Random pair similarity (baseline) | 0.092 | 0.243 | 0.157 |
| **Delta (morphological signal)** | +0.025 | **+0.515** | **+0.467** |
| Usable pairs | 11 | 2,000 | 2,000 |

**Key finding:** BPE and SentencePiece embeddings capture morphological relationships 10–20x better than word-level across both configurations. The delta between morphological and random similarity is ~0.5 for BPE and ~0.45 for SP, but only ~0.03–0.05 for word-level. Because "sport" and "sporting" share subword pieces, their embeddings are naturally close in BPE/SP models. The word tokenizer treats them as unrelated tokens.

**Suffix vs. prefix asymmetry (SentencePiece):**
- Config C: Suffix pairs mean cosine similarity **0.706**, Prefix pairs **0.394**
- Config D: Suffix pairs **0.772**, Prefix pairs **0.454**

This asymmetry occurs because Unigram tokenization tends to keep word stems intact and split off suffixes (e.g., "play" + "ing"), so suffixed forms share more subword material with their base. Prefixed forms (e.g., "un" + "nest") share less overlap. BPE showed a more balanced pattern (suffix ~0.75, prefix ~0.72) in both configurations.

### 3.5 Vocabulary Coverage

**Config C (18K vocab):**

| Benchmark | Word Usable | BPE Usable | SP Usable |
|---|---|---|---|
| Frequent words | 40 | 39 | 29 |
| Medium-frequency | 400 | 400 (390 single / 10 multi) | 392 (151 single / 241 multi) |
| WikiText rare | **166** | 500 (11 single / 489 multi) | 485 (82 single / 403 multi) |
| Shakespeare rare | 258 | 494 | 463 |
| MorphyNet | 444 | 500 | 493 |
| Embedding pairs | 22 | 2,000 | 2,000 |

**Config D (8K vocab):**

| Benchmark | Word Usable | BPE Usable | SP Usable |
|---|---|---|---|
| Frequent words | 40 | 39 | 29 |
| Medium-frequency | 356 | 400 (24 single / 376 multi) | 383 (140 single / 243 multi) |
| WikiText rare | **0** | 500 | 450 |
| Shakespeare rare | 171 | 494 | 418 |
| MorphyNet | 231 | 500 | 464 |
| Embedding pairs | 11 | 2,000 | 2,000 |

The 18K vocabulary (Config C) provides substantially better coverage for the word tokenizer across all benchmarks. Notably, BPE has far more single-token words with 18K vocab (390 vs. 24 on medium-frequency), meaning fewer words need multi-token reconstruction. This reflects a general principle: larger vocabularies produce less subword fragmentation.

---

## 4. What Changes Produced Results?

| Change | Impact |
|---|---|
| Fixing eval bugs (string/ID, piece accuracy, coverage) | Fixed measurement — no accuracy change |
| Increasing training from 5K steps to 20 epochs | Models converged properly but still 0% top-k |
| Adding validation loss tracking | Confirmed convergence, revealed overfitting |
| **Increasing hidden 256 → 512** | **Dominant factor** — word perplexity 337 → 30, first meaningful predictions |
| Reducing vocab 18K → 8K | Modest additional improvement — perplexity 30 → 22, but eliminates rare word coverage |
| Adding top-50/top-100 metrics | Revealed accuracy that top-10 missed |
| Decoupling eval filters from word vocab | Enabled rare word evaluation for BPE/SP |
| Adding embedding similarity | Revealed subword advantage in morphological representation |

**The ablation study revealed that increasing hidden size was the dominant factor.** Vocab reduction alone (Config B: 8K/256) barely improved results (0.3% top-10 on medium words). Hidden size increase alone (Config C: 18K/512) produced the full breakthrough (16.3% top-10 on medium words, 7.2% top-10 on rare words). Combining both changes (Config D: 8K/512) provided marginal additional improvement on in-vocabulary words but eliminated rare word coverage entirely.

**Config C (18K/512) emerged as the strongest overall configuration** because it achieves comparable prediction accuracy to Config D while retaining the ability to evaluate rare words — the project's central research question.

---

## 5. Potential Future Directions

1. **Larger training corpus (WikiText-103):** A v2 notebook using WikiText-103 (50x more data) was created but not fully evaluated due to Colab timeouts. Running this on an HPC cluster could significantly improve BPE and SP performance, as subword models generally benefit more from additional data.

2. **Shakespeare as training corpus:** The original proposal specified Mini-Shakespeare as the training database. Training on Shakespeare and evaluating on WikiText-2 would test domain-transfer effects and more closely match the proposal. A comparison between WikiText-trained and Shakespeare-trained models would strengthen the analysis.

3. **Vocab size sweep:** Testing vocab sizes of 5K, 8K, 10K, 15K, and 18K would map the accuracy-vs-coverage trade-off curve. There likely exists a sweet spot where the word tokenizer still has rare word coverage while maintaining prediction advantage.

4. **More BPE training:** BPE validation loss was still decreasing at epoch 20 (6.13), unlike word and SP which had plateaued. Additional epochs or a learning rate adjustment could close the gap.

5. **Character-level BPE vs. byte-level BPE:** The current BPE uses byte-level encoding (GPT-2 style), which produces very fragmented subword pieces at 8K vocab. A character-level BPE might produce more meaningful subword units and improve BPE performance.

6. **Contextual embedding similarity:** The current embedding analysis uses static input embeddings (the embedding layer weights). Extracting contextual embeddings (hidden states from the transformer layers) for words in context could reveal richer morphological structure.

7. **Multiple random seeds:** Running with 3–5 different seeds and reporting mean ± standard deviation would strengthen statistical validity, as proposed in the original project plan.

---

## 6. Key Insights

### The Central Finding: Tokenization Creates a Prediction-Representation Trade-off

Word-level tokenization and subword tokenization embody fundamentally different inductive biases:

- **Word-level** concentrates all learning on whole-word prediction. Each word gets its own embedding, and the model learns to predict from a closed vocabulary. This produces strong predictions for in-vocabulary words (25% top-10 on frequent words) but completely fails on rare or unseen words (0 usable examples on WikiText rare words).

- **Subword tokenization** distributes learning across shared morphological pieces. "Playing," "played," and "player" share the subword "play," creating natural morphological structure in embedding space (cosine similarity 0.76 for BPE, 0.62 for SP between related forms). However, predicting the correct sequence of subword pieces is a harder task than predicting a single whole word, resulting in near-zero prediction accuracy on a small model.

This trade-off is the **inductive bias** referenced in the project title. The choice of tokenizer doesn't just change how text is segmented — it fundamentally changes what the model can learn and what it can represent.

### Secondary Findings

1. **SentencePiece (Unigram) produces more linguistically meaningful subword units than byte-level BPE.** SP consistently achieves higher piece-level reconstruction accuracy (7.2% vs. 0.0% on WikiText rare in Config C; 4.4% vs. 0.0% on Shakespeare) and shows a meaningful suffix/prefix asymmetry in embedding similarity that reflects real morphological structure.

2. **Hidden size dominates over vocabulary reduction.** The ablation study shows that increasing hidden size from 256 to 512 accounts for nearly all the improvement. Config C (18K/512) achieves 32.5% top-10 on frequent words while Config B (8K/256) achieves 0.0%. Vocab reduction provides marginal additional benefit only when combined with the larger model.

3. **Vocabulary size creates a coverage-accuracy trade-off, not a monotonic improvement.** Reducing vocab from 18K to 8K slightly improves word tokenizer accuracy on in-vocabulary words (by reducing the prediction space) but eliminates coverage of rare words entirely. The 18K vocab configuration can evaluate 166 rare words; the 8K configuration can evaluate zero. For the project's research question about rare words, the larger vocabulary is essential.

4. **The OOV problem is not theoretical — it is directly observable and quantifiable.** Comparing Configs C and D shows the trade-off precisely: Config D (8K vocab) has zero usable rare word examples for the word tokenizer, while Config C (18K vocab) has 166 usable examples with 7.2% top-10 accuracy. Meanwhile, BPE and SP maintain 485–500 usable examples regardless of vocab size because they decompose all words into subword pieces.

5. **SentencePiece benefits most from increased model capacity.** SP's validation loss improved from 5.79 (Config A) to 5.05 (Config C) with the hidden size increase — a perplexity reduction from ~326 to ~156. Its reconstruction piece accuracy jumped from 0.9% to 7.2% on WikiText rare words. This suggests that SP's linguistically meaningful subword units can be better exploited by a larger transformer.

---

## 7. Alignment with Project Proposal and Feedback

### Proposal Requirements

| Requirement | Status | Notes |
|---|---|---|
| Three tokenization strategies compared | **Met** | Word, BPE, SentencePiece (Unigram) |
| Identical architecture across models | **Met** | 4-layer, 512 hidden — 22.2M params (18K vocab) or 17.1M (8K vocab) |
| Same vocab size for fair comparison | **Met** | All three tokenizers share the same vocab size within each configuration |
| Same training budget | **Met** | 20 epochs each on WikiText-2 |
| Explicit OOV/UNK policy | **Met** | Word-level maps unseen words to `<unk>` |
| HuggingFace tokenizer implementations | **Met** | HuggingFace `tokenizers` for BPE, `sentencepiece` library for SP |
| Masked rare word prediction | **Met** | Evaluated across WikiText, Shakespeare, MorphyNet |
| Closest neighbors to morphed words | **Met** | Embedding similarity evaluation with cosine similarity on MorphyNet pairs |
| WikiText-2 for evaluation | **Met** | Rare-word contexts from WikiText-2 validation |
| MorphyNet for morphological testing | **Met** | Both prediction and embedding similarity evaluation |
| Results as percentage of correct predictions | **Met** | Top-1/5/10/50/100 accuracy reported |
| Results sorted by word length, sentence length | **Met** | Breakdowns by word length, subword piece count, frequency |
| Multiple runs for aggregated scores | **Partially met** | Single seed, but ablation study provides 4 configurations for comparison |

### Deviations from Proposal (with Justification)

1. **Training corpus:** The proposal specified Mini-Shakespeare as the training database. We switched to WikiText-2 because Shakespeare's archaic language produced poor performance on modern evaluation text. WikiText-2 provides contemporary English that better supports learning of the vocabulary needed for our evaluation benchmarks. Shakespeare data is still used as a domain-shift evaluation set.

2. **WordNet not used:** The proposal mentioned using WordNet to find words with few related neighbors. In practice, WikiText-2's frequency-based rare word selection (freq 1–9 in training data) serves the same purpose — identifying words with sparse training signal. The frequency-based approach is more directly interpretable and avoids the complexity of WordNet's semantic network.

3. **Model size adjusted:** The proposal specified "4-layer Transformer, 256 hidden size." We increased hidden size to 512 after the original 256-hidden models produced zero predictive accuracy across all tokenizers. The architecture change was necessary to produce any meaningful results for comparison. Importantly, the same architecture is used for all three tokenizers, preserving the fair comparison that is the project's core design principle.

4. **Vocabulary explored at both 18K and 8K:** The ablation study tested both sizes. The 8K vocabulary demonstrated the OOV trade-off (better in-vocabulary accuracy but zero rare word coverage), while 18K vocabulary with the larger model proved to be the stronger configuration overall, enabling rare word evaluation across all tokenizers.

### Professor's Feedback

**1. "Make sure to have a proper discussion of relevant prior work."**

Our references provide strong foundations: Sennrich et al. (2016) for BPE's origin in neural machine translation of rare words (directly relevant to our rare-word focus), Kudo (2018) for Unigram subword regularization, Kudo & Richardson (2018) for SentencePiece as a language-independent framework, and Bengio et al. (2003) for the word-level neural LM baseline. Our final report should discuss how each paper's findings connect to our observed results — particularly how Sennrich et al.'s finding that BPE improves rare word translation aligns with our finding that BPE's morphological embeddings capture relationships that word-level cannot, even though word-level achieves higher prediction accuracy on in-vocabulary words.

**2. "Make sure to have a thorough evaluation since the code is similar to what we have already discussed in class in demos or assignments."**

Our evaluation goes substantially beyond a standard MLM training exercise:
- **Five evaluation benchmarks:** frequent, medium-frequency, rare (WikiText), rare (Shakespeare domain-shift), and morphological (MorphyNet)
- **Three evaluation paradigms:** single-token prediction (top-k accuracy, MRR, mean rank), multi-token reconstruction (piece accuracy, exact match), and embedding similarity (cosine similarity with random baseline)
- **Four-configuration ablation study:** systematically isolating the contribution of vocabulary size and model capacity
- **Multiple analysis dimensions:** breakdowns by word frequency, word length, subword piece count, affix type (prefix vs. suffix), and cross-domain comparison
- **Coverage analysis:** demonstrating the OOV problem quantitatively through usable-example counts per tokenizer
- **Fair comparison methodology:** unified evaluation that handles single-token and multi-token words consistently across tokenizers

The depth of evaluation — particularly the ablation study, embedding similarity analysis, and the multi-dimensional breakdowns — differentiates this from a standard class demo.

---

## 8. Summary of Notebook Versions

| Version | File | Config | Key Changes |
|---|---|---|---|
| v1 (original) | `GenAI_Group_Project_mediumfreq_softmetrics (1).ipynb` | 18K/256 | Original implementation. 5K steps. |
| v1 results | `GenAI_Group_Project_mediumfreq_softmetrics_results.ipynb` | 18K/256 | Bug fixes + 20 epochs. 0% accuracy. |
| v2 (WikiText-2) | `GenAI_Group_Project_v2_wikitext2.ipynb` | 8K/512 | First non-zero results. |
| v2 (WikiText-103) | `GenAI_Group_Project_v2_wikitext103.ipynb` | 8K/512 | 50x more data. Not fully evaluated. |
| **v3** | **`GenAI_Group_Project_v3.ipynb`** | **8K/512** | **Decoupled eval filters + embedding similarity.** |
| v3 results | `GenAI_Group_Project_v3_results2.ipynb` | 8K/512 | Full results (Config D). |
| **Ablation B** | `GenAI_Group_Project_ablation_vocab8k_hidden256.ipynb` | **8K/256** | **Vocab reduction only.** |
| Ablation B results | `GenAI_Group_Project_ablation_vocab8k_hidden256_results.ipynb` | 8K/256 | Complete results. |
| **Ablation C** | `GenAI_Group_Project_ablation_vocab18k_hidden512.ipynb` | **18K/512** | **Hidden size increase only.** |
| Ablation C results | `GenAI_Group_Project_ablation_vocab18k_hidden512_results.ipynb` | 18K/512 | Complete results. Best overall config. |
