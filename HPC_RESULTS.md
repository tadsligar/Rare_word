# HPC Experimental Results — Full Ablation Study

All results were produced by running SLURM batch jobs on the Auburn HPC cluster (1× GPU, 64 GB RAM) using the Python scripts converted from the project notebooks. This document captures every numeric metric from all 7 runs, including the full top-N accuracy tables that were truncated in standard pandas output.

---

## Experiment Overview

| Config | Dataset | Vocab | Hidden | Heads | Intermediate | Parameters | Epochs | Script |
|--------|---------|-------|--------|-------|-------------|------------|--------|--------|
| **A** 8K/256 | WikiText-2 | 8 000 | 256 | 4 | 1 024 | 5 347 904 | 20 | ablation_vocab8k_hidden256 |
| **B** 8K/512 | WikiText-2 | 8 000 | 512 | 8 | 2 048 | 17 110 336 | 20 | v3 |
| **C** 18K/512 | WikiText-2 | 18 000 | 512 | 8 | 2 048 | 22 240 336 | 20 | ablation_vocab18k_hidden512 |
| **A** 8K/256 | WikiText-103 | 8 000 | 256 | 4 | 1 024 | 5 347 904 | 3 | ablation_vocab8k_hidden256_wikitext103 |
| **B** 8K/512 | WikiText-103 | 8 000 | 512 | 8 | 2 048 | 17 110 336 | 3 | v3_wikitext103 |
| **C** 18K/512 | WikiText-103 | 18 000 | 512 | 8 | 2 048 | 22 240 336 | 3 | ablation_vocab18k_hidden512_wikitext103 |
| **D** 18K/768 | WikiText-103 | 18 000 | 768 | 12 | 3 072 | 42 985 296 | 3 | ablation_vocab18k_hidden768_wikitext103 |

> **Note on epochs:** WikiText-103 has ~50× more text than WikiText-2. 3 epochs on WT103 ≈ 150 epochs on WT2 by token count, making training times comparable (3–10 hours per tokenizer).

Three tokenizers are trained and evaluated independently for each config:
- **Word** — regex-based word-level tokenization, vocabulary built from training corpus
- **BPE** — byte-pair encoding (HuggingFace tokenizers)
- **SP** — SentencePiece Unigram

---

## 1. Final Training Losses

Training loss is reported at the end of the final epoch. Each tokenizer trains its own independent BERT-style masked language model.

| Config | Dataset | Tokenizer | Final Train Loss | Epochs |
|--------|---------|-----------|-----------------|--------|
| A 8K/256 | WikiText-2 | Word | 5.624 | 20 |
| A 8K/256 | WikiText-2 | BPE | 6.301 | 20 |
| A 8K/256 | WikiText-2 | SP | 5.746 | 20 |
| B 8K/512 | WikiText-2 | Word | 3.759 | 20 |
| B 8K/512 | WikiText-2 | BPE | 6.181 | 20 |
| B 8K/512 | WikiText-2 | SP | 5.632 | 20 |
| C 18K/512 | WikiText-2 | Word | 4.060 | 20 |
| C 18K/512 | WikiText-2 | BPE | 6.249 | 20 |
| C 18K/512 | WikiText-2 | SP | 5.680 | 20 |
| A 8K/256 | WikiText-103 | Word | 3.569 | 3 |
| A 8K/256 | WikiText-103 | BPE | 4.534 | 3 |
| A 8K/256 | WikiText-103 | SP | 2.879 | 3 |
| B 8K/512 | WikiText-103 | Word | 3.015 | 3 |
| B 8K/512 | WikiText-103 | BPE | 4.114 | 3 |
| B 8K/512 | WikiText-103 | SP | 3.848 | 3 |
| C 18K/512 | WikiText-103 | Word | 3.241 | 3 |
| C 18K/512 | WikiText-103 | BPE | 4.508 | 3 |
| C 18K/512 | WikiText-103 | SP | 4.117 | 3 |
| D 18K/768 | WikiText-103 | Word | 3.352 | 3 |
| D 18K/768 | WikiText-103 | BPE | 4.190 | 3 |
| D 18K/768 | WikiText-103 | SP | 4.948 | 3 |

**Key observations:**
- Word tokenizer achieves significantly lower loss than BPE/SP on WikiText-2 (word-level task is easier at that scale)
- WikiText-103 dramatically reduces word loss (3.0–3.6 vs 3.8–5.6), evidence of much better training signal
- BPE/SP losses on WT103 converge more slowly — their larger effective vocabularies require more data
- Config D (18K/768) shows slightly higher SP loss than C (18K/512), suggesting larger models need more epochs to converge on SP
- Config A (8K/256) achieves anomalously low SP loss on WT103 (2.879) — likely because the small vocab forces aggressive subword merging that reduces effective sequence complexity

---

## 2. Embedding Similarity — Morphological Pairs

The model is evaluated by computing cosine similarity between the embeddings of morphologically related word pairs (e.g., "happy"→"happiness", "run"→"running") from the MorphyNet dataset. A random baseline similarity is computed from the same model to account for the general "similarity floor" of the embedding space.

### 2a. Summary Table

| Config | Dataset | Tokenizer | Pairs | Mean Sim | Random Baseline | **Delta** |
|--------|---------|-----------|-------|----------|----------------|-----------|
| A 8K/256 | WikiText-2 | Word | 10 | 0.1117 | −0.1014 | **+0.2131** |
| A 8K/256 | WikiText-2 | BPE | 2 000 | 0.5765 | 0.1789 | **+0.3976** |
| A 8K/256 | WikiText-2 | SP | 2 000 | 0.6816 | 0.2769 | **+0.4047** |
| B 8K/512 | WikiText-2 | Word | 10 | 0.1518 | −0.0577 | **+0.2095** |
| B 8K/512 | WikiText-2 | BPE | 2 000 | 0.5821 | 0.2041 | **+0.3780** |
| B 8K/512 | WikiText-2 | SP | 2 000 | 0.7013 | 0.3356 | **+0.3657** |
| C 18K/512 | WikiText-2 | Word | 20 | 0.2139 | 0.0057 | **+0.2082** |
| C 18K/512 | WikiText-2 | BPE | 2 000 | 0.6857 | 0.3336 | **+0.3521** |
| C 18K/512 | WikiText-2 | SP | 2 000 | 0.7316 | 0.3621 | **+0.3695** |
| A 8K/256 | WikiText-103 | Word | 12 | 0.4045 | 0.1377 | **+0.2668** |
| A 8K/256 | WikiText-103 | BPE | 2 000 | 0.7978 | 0.3387 | **+0.4591** |
| A 8K/256 | WikiText-103 | SP | 2 000 | 0.8183 | 0.5511 | **+0.2672** |
| B 8K/512 | WikiText-103 | Word | 12 | 0.3981 | 0.1453 | **+0.2528** |
| B 8K/512 | WikiText-103 | SP | 2 000 | 0.8983 | 0.7458 | **+0.1525** |
| B 8K/512 | WikiText-103 | BPE | 2 000 | 0.8080 | 0.3765 | **+0.4314** |
| C 18K/512 | WikiText-103 | Word | 20 | 0.4135 | 0.2578 | **+0.1556** |
| C 18K/512 | WikiText-103 | BPE | 2 000 | 0.8333 | 0.5051 | **+0.3282** |
| C 18K/512 | WikiText-103 | SP | 2 000 | 0.7881 | 0.5261 | **+0.2620** |
| D 18K/768 | WikiText-103 | Word | 20 | 0.4686 | 0.3802 | **+0.0884** |
| D 18K/768 | WikiText-103 | BPE | 2 000 | 0.8579 | 0.5831 | **+0.2748** |
| D 18K/768 | WikiText-103 | SP | 2 000 | 0.8146 | 0.6071 | **+0.2075** |

### 2b. By Affix Type (Prefix vs Suffix)

| Config | Dataset | Tokenizer | Affix | Pairs | Mean Sim |
|--------|---------|-----------|-------|-------|----------|
| A 8K/256 | WikiText-2 | Word | suffix | 4 | 0.1224 |
| A 8K/256 | WikiText-2 | Word | prefix | 6 | 0.1015 |
| A 8K/256 | WikiText-2 | BPE | suffix | 1 069 | 0.5864 |
| A 8K/256 | WikiText-2 | BPE | prefix | 931 | 0.5643 |
| A 8K/256 | WikiText-2 | SP | suffix | 1 069 | 0.6982 |
| A 8K/256 | WikiText-2 | SP | prefix | 931 | 0.6629 |
| B 8K/512 | WikiText-2 | Word | suffix | 4 | 0.1459 |
| B 8K/512 | WikiText-2 | Word | prefix | 6 | 0.1578 |
| B 8K/512 | WikiText-2 | BPE | suffix | 1 069 | 0.5888 |
| B 8K/512 | WikiText-2 | BPE | prefix | 931 | 0.5737 |
| B 8K/512 | WikiText-2 | SP | suffix | 1 069 | 0.7106 |
| B 8K/512 | WikiText-2 | SP | prefix | 931 | 0.6902 |
| C 18K/512 | WikiText-2 | Word | suffix | 13 | 0.2460 |
| C 18K/512 | WikiText-2 | Word | prefix | 7 | 0.1527 |
| C 18K/512 | WikiText-2 | BPE | suffix | 1 069 | 0.6944 |
| C 18K/512 | WikiText-2 | BPE | prefix | 931 | 0.6733 |
| C 18K/512 | WikiText-2 | SP | suffix | 1 069 | 0.7410 |
| C 18K/512 | WikiText-2 | SP | prefix | 931 | 0.7205 |
| A 8K/256 | WikiText-103 | Word | suffix | 8 | 0.4110 |
| A 8K/256 | WikiText-103 | Word | prefix | 4 | 0.3914 |
| A 8K/256 | WikiText-103 | BPE | suffix | 1 069 | 0.8047 |
| A 8K/256 | WikiText-103 | BPE | prefix | 931 | 0.7899 |
| A 8K/256 | WikiText-103 | SP | suffix | 1 069 | 0.8697 |
| A 8K/256 | WikiText-103 | SP | prefix | 931 | 0.7592 |
| B 8K/512 | WikiText-103 | Word | suffix | 8 | 0.3954 |
| B 8K/512 | WikiText-103 | Word | prefix | 4 | 0.4036 |
| B 8K/512 | WikiText-103 | BPE | suffix | 1 069 | 0.8136 |
| B 8K/512 | WikiText-103 | BPE | prefix | 931 | 0.8015 |
| B 8K/512 | WikiText-103 | SP | suffix | 1 069 | 0.9288 |
| B 8K/512 | WikiText-103 | SP | prefix | 931 | 0.8633 |
| C 18K/512 | WikiText-103 | Word | suffix | 15 | 0.4100 |
| C 18K/512 | WikiText-103 | Word | prefix | 5 | 0.4237 |
| C 18K/512 | WikiText-103 | BPE | suffix | 1 069 | 0.8392 |
| C 18K/512 | WikiText-103 | BPE | prefix | 931 | 0.8265 |
| C 18K/512 | WikiText-103 | SP | suffix | 1 069 | 0.8570 |
| C 18K/512 | WikiText-103 | SP | prefix | 931 | 0.7089 |
| D 18K/768 | WikiText-103 | Word | suffix | 15 | 0.4592 |
| D 18K/768 | WikiText-103 | Word | prefix | 5 | 0.4970 |
| D 18K/768 | WikiText-103 | BPE | suffix | 1 069 | 0.8618 |
| D 18K/768 | WikiText-103 | BPE | prefix | 931 | 0.8534 |
| D 18K/768 | WikiText-103 | SP | suffix | 1 069 | 0.8744 |
| D 18K/768 | WikiText-103 | SP | prefix | 931 | 0.7460 |

**Key observations:**
- Suffix relationships are consistently captured better than prefix relationships by SP (consistent with SP's suffix-biased segmentation for English morphology)
- BPE shows smaller prefix/suffix gap than SP
- WT103 dramatically raises both the mean similarity AND the random baseline — the absolute delta is a more informative comparison
- Config D (18K/768) has the highest raw BPE mean sim (0.8579) but lower delta (+0.2748) because the random baseline is also higher (0.5831)
- Config A (8K/256) on WT103 achieves the highest BPE delta (+0.4591) — the smaller vocabulary may force more meaningful subword clustering

---

## 3. Benchmark Evaluations

For each benchmark, models fill in a masked word `[MASK]` and are ranked by the log-probability of the gold answer. "Usable" means the gold word is representable by that tokenizer.

> **Note on top1/top5 visibility:** Pandas truncated columns 4–5 (top1, top5) in runs where there were too many columns to display. Full top10/50/100 values are always captured. Top1 and top5 are shown where visible; otherwise noted as not directly logged.

---

### 3a. Frequent Content-Word Benchmark

Evaluates how well models predict common content words (high-frequency, in-vocabulary for all tokenizers). Only 1 example was selected for WikiText-103 configs (small Shakespeare "frequent" intersection), while WikiText-2 configs had 40 usable examples.

#### WikiText-103 (1 example)

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| D 18K/768 | Word | 1 | 1.000 | 1.0 | 1.0 | 1.0 |
| D 18K/768 | BPE | 1 | 0.167 | 1.0 | 1.0 | 1.0 |
| D 18K/768 | SP | 1 | 0.063 | 0.0 | 1.0 | 1.0 |
| C 18K/512 | Word | 1 | 1.000 | 1.0 | 1.0 | 1.0 |
| C 18K/512 | BPE | 1 | 0.333 | 1.0 | 1.0 | 1.0 |
| C 18K/512 | SP | 1 | 1.000 | 1.0 | 1.0 | 1.0 |
| A 8K/256 | Word | 1 | 0.043 | 0.0 | 1.0 | 1.0 |
| A 8K/256 | BPE | 1 | 0.050 | 0.0 | 1.0 | 1.0 |
| A 8K/256 | SP | 1 | 0.071 | 0.0 | 1.0 | 1.0 |
| B 8K/512 | Word | 1 | 0.333 | 1.0 | 1.0 | 1.0 |
| B 8K/512 | BPE | 1 | 1.000 | 1.0 | 1.0 | 1.0 |
| B 8K/512 | SP | 1 | 0.100 | 1.0 | 1.0 | 1.0 |

> With only 1 example this is illustrative only. All models find the correct answer in the top 100 (or top 50 for SP/18K/768).

#### WikiText-2 (40 examples for word, ~39/29 for BPE/SP)

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| C 18K/512 | Word | 40 | 0.0816 | 0.275 | 0.525 | 0.600 |
| C 18K/512 | BPE | 39 | 0.0033 | 0.000 | 0.026 | 0.051 |
| C 18K/512 | SP | 29 | 0.0092 | 0.000 | 0.103 | 0.276 |
| B 8K/512 | Word | 40 | 0.0726 | 0.175 | 0.425 | 0.475 |
| B 8K/512 | BPE | 39 | 0.0037 | 0.000 | 0.026 | 0.128 |
| B 8K/512 | SP | 29 | 0.0109 | 0.000 | 0.138 | 0.241 |
| A 8K/256 | Word | 40 | 0.0053 | 0.000 | 0.050 | 0.200 |
| A 8K/256 | BPE | 39 | 0.0023 | 0.000 | 0.000 | 0.026 |
| A 8K/256 | SP | 29 | 0.0067 | 0.000 | 0.103 | 0.207 |

**Observations:** Word tokenizer is the clear winner on frequent content words for WT2, with Config C (18K/512) achieving the best MRR (0.082) and Top-100 (60%). BPE struggles significantly on frequent words — its predictions rank the true word far down. SP does better than BPE at Top-100 due to its unigram model properties.

---

### 3b. WikiText Medium-Frequency Content-Word Benchmark

Tests prediction of medium-frequency words from WikiText validation (400 examples sampled). These words appear 50–1000× in the training set.

> **Critical note:** For all WikiText-103 configs, the word tokenizer gets **0 usable examples** — medium-frequency WikiText words are almost entirely absent from the word-level vocabulary (which was built from Shakespeare). This is a fundamental limitation of word-level tokenization when training/evaluation domains differ.

#### WikiText-2 Configs

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| C 18K/512 | Word | 400 | 0.0934 | 0.180 | 0.308 | 0.418 |
| C 18K/512 | BPE | 390 | 0.0017 | 0.000 | 0.015 | 0.033 |
| C 18K/512 | SP | 151 | 0.0018 | 0.000 | 0.000 | 0.020 |
| B 8K/512 | Word | 356 | 0.0889 | 0.163 | 0.317 | 0.424 |
| B 8K/512 | BPE | 24 | 0.0026 | 0.000 | 0.042 | 0.042 |
| B 8K/512 | SP | 140 | 0.0024 | 0.000 | 0.014 | 0.036 |
| A 8K/256 | Word | 356 | 0.0023 | 0.000 | 0.020 | 0.042 |
| A 8K/256 | BPE | 24 | 0.0006 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | SP | 140 | 0.0017 | 0.000 | 0.000 | 0.029 |

#### WikiText-103 Configs (word tokenizer: 0 usable examples for all)

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| D 18K/768 | BPE | 1 | 0.000089 | 0.0 | 0.0 | 0.0 |
| D 18K/768 | SP | 3 | 0.000654 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | BPE | 1 | 0.000155 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | SP | 3 | 0.015354 | 0.0 | 0.333 | 0.333 |
| A 8K/256 | BPE | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A 8K/256 | SP | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B 8K/512 | BPE | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B 8K/512 | SP | 0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Observations:** Config C (18K/512) on WT2 is the top performer with 42% top-100 accuracy and 18% top-10 accuracy for medium-frequency words using the word tokenizer. Config B achieves nearly identical results. Config A (8K/256) fails almost completely — its smaller hidden dimension appears to be the bottleneck (MRR drops 40×). WT103 configs have essentially no usable examples for the word tokenizer, and BPE/SP usable counts are near-zero due to vocabulary coverage differences.

---

### 3c. WikiText Rare-Word Single-Token Benchmark

Tests prediction of words appearing fewer than 5× in the WikiText training set, considering only examples where the gold word maps to a single token (most challenging evaluation).

#### WikiText-2 Configs

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| C 18K/512 | Word | 166 | 0.0491 | 0.090 | 0.157 | 0.211 |
| C 18K/512 | BPE | 11 | 0.0002 | 0.000 | 0.000 | 0.000 |
| C 18K/512 | SP | 82 | 0.0007 | 0.000 | 0.012 | 0.012 |
| B 8K/512 | Word | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| B 8K/512 | BPE | 1 | 0.0004 | 0.000 | 0.000 | 0.000 |
| B 8K/512 | SP | 19 | 0.0014 | 0.000 | 0.000 | 0.053 |
| A 8K/256 | Word | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | BPE | 1 | 0.0005 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | SP | 19 | 0.0010 | 0.000 | 0.000 | 0.000 |

> Config C (18K vocab) is the only one where the word tokenizer gets usable single-token rare-word examples (166), achieving 9% top-10 and 21% top-100. The 8K configs have too small a vocabulary to represent these rare words as single tokens.

#### WikiText-103 Configs (effectively all zeros)

| Config | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| D 18K/768 | Word | 0 | 0.000 | 0.0 | 0.0 | 0.0 |
| D 18K/768 | BPE | 1 | 0.000138 | 0.0 | 0.0 | 0.0 |
| D 18K/768 | SP | 0 | 0.000 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | Word | 0 | 0.000 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | BPE | 1 | 0.000110 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | SP | 0 | 0.000 | 0.0 | 0.0 | 0.0 |
| A 8K/256 | All | 0 | 0.000 | 0.0 | 0.0 | 0.0 |
| B 8K/512 | All | 0 | 0.000 | 0.0 | 0.0 | 0.0 |

---

### 3d. WikiText Rare-Word Unified Evaluation (Single + Multi-Token)

The unified evaluation allows multi-token rare words by reconstructing the predicted sequence from multiple mask positions.

#### WikiText-2 Configs

| Config | Tokenizer | Usable | Single-Tok | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|------------|-------|-------|--------|
| C 18K/512 | Word | 166 | 166 | 0.090 | 0.157 | 0.211 |
| C 18K/512 | BPE | 500 | 11 | 0.000 | 0.000 | 0.000 |
| C 18K/512 | SP | 485 | 82 | 0.000 | 0.002 | 0.002 |
| B 8K/512 | Word | 0 | 0 | 0.000 | 0.000 | 0.000 |
| B 8K/512 | BPE | 500 | 1 | 0.000 | 0.000 | 0.000 |
| B 8K/512 | SP | 450 | 19 | 0.000 | 0.000 | 0.002 |
| A 8K/256 | Word | 0 | 0 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | BPE | 500 | 1 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | SP | 450 | 19 | 0.000 | 0.000 | 0.000 |

#### WikiText-103 Configs (all zeros across the board)

| Config | Tokenizer | Usable | Single-Tok | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|------------|-------|-------|--------|
| D 18K/768 | Word | 0 | 0 | 0.0 | 0.0 | 0.0 |
| D 18K/768 | BPE | 500 | 1 | 0.0 | 0.0 | 0.0 |
| D 18K/768 | SP | 481 | 0 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | Word | 0 | 0 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | BPE | 500 | 1 | 0.0 | 0.0 | 0.0 |
| C 18K/512 | SP | 481 | 0 | 0.0 | 0.002 | 0.003 |
| A 8K/256 | BPE | 500 | 0 | 0.0 | 0.0 | 0.0 |
| A 8K/256 | SP | 477 | 0 | 0.0 | 0.0 | 0.0 |
| B 8K/512 | BPE | 500 | 0 | 0.0 | 0.0 | 0.0 |
| B 8K/512 | SP | 477 | 0 | 0.0 | 0.0 | 0.0 |

> The WT103 models completely fail to predict WikiText rare words — these models were trained on WT103 data but the rare words in the evaluation are WT103-specific. This is counterintuitive and likely reflects that 3 epochs on WT103 is still insufficient to build confident rare-word predictions, despite the large data volume.

---

### 3e. Shakespeare Rare-Word Single-Token Benchmark

Shakespeare contains archaic words not present in WikiText. This cross-domain test evaluates generalization. 500 rare words are sampled; "usable" counts vary by tokenizer vocabulary coverage.

| Config | Dataset | Tokenizer | Usable | MRR | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|-----|-------|-------|--------|
| D 18K/768 | WT103 | Word | 262 | 0.01430 | 0.02290 | 0.06489 | 0.10305 |
| D 18K/768 | WT103 | BPE | 196 | 0.00036 | 0.00000 | 0.00000 | 0.00000 |
| D 18K/768 | WT103 | SP | 212 | 0.00111 | 0.00000 | 0.00472 | 0.02358 |
| C 18K/512 | WT103 | Word | 262 | 0.00752 | 0.01145 | 0.08779 | 0.11832 |
| C 18K/512 | WT103 | BPE | 196 | 0.00417 | 0.00510 | 0.02041 | 0.04082 |
| C 18K/512 | WT103 | SP | 212 | 0.00113 | 0.00000 | 0.00472 | 0.02830 |
| A 8K/256 | WT103 | Word | 180 | 0.01408 | 0.04444 | 0.11111 | 0.16667 |
| A 8K/256 | WT103 | BPE | 87 | 0.00686 | 0.02299 | 0.05747 | 0.10345 |
| A 8K/256 | WT103 | SP | 121 | 0.00753 | 0.01653 | 0.04132 | 0.10744 |
| B 8K/512 | WT103 | Word | 180 | 0.01065 | 0.02222 | 0.09444 | 0.17222 |
| B 8K/512 | WT103 | BPE | 87 | 0.00614 | 0.01149 | 0.03448 | 0.05747 |
| B 8K/512 | WT103 | SP | 121 | 0.00156 | 0.00000 | 0.01653 | 0.01653 |
| C 18K/512 | WT2 | Word | 258 | 0.00507 | 0.00775 | 0.02713 | 0.05814 |
| C 18K/512 | WT2 | BPE | 198 | 0.00050 | 0.00000 | 0.00000 | 0.00505 |
| C 18K/512 | WT2 | SP | 145 | 0.00079 | 0.00000 | 0.00000 | 0.00690 |
| B 8K/512 | WT2 | Word | 171 | 0.00626 | 0.00585 | 0.05848 | 0.10526 |
| B 8K/512 | WT2 | BPE | 94 | 0.00104 | 0.00000 | 0.01064 | 0.01064 |
| B 8K/512 | WT2 | SP | 114 | 0.00105 | 0.00000 | 0.00877 | 0.00877 |
| A 8K/256 | WT2 | Word | 171 | 0.00102 | 0.00000 | 0.00585 | 0.01170 |
| A 8K/256 | WT2 | BPE | 94 | 0.00077 | 0.00000 | 0.00000 | 0.00000 |
| A 8K/256 | WT2 | SP | 114 | 0.00122 | 0.00000 | 0.01754 | 0.01754 |

---

### 3f. Shakespeare Rare-Word Unified Evaluation

| Config | Dataset | Tokenizer | Usable | Single-Tok | Top10 | Top50 | Top100 |
|--------|---------|-----------|---------|------------|-------|-------|--------|
| D 18K/768 | WT103 | Word | 262 | 262 | 0.02290 | 0.06489 | 0.10305 |
| D 18K/768 | WT103 | BPE | 494 | 196 | 0.00000 | 0.00000 | 0.00000 |
| D 18K/768 | WT103 | SP | 479 | 212 | 0.00000 | 0.00209 | 0.01044 |
| C 18K/512 | WT103 | Word | 262 | 262 | 0.01145 | 0.08779 | 0.11832 |
| C 18K/512 | WT103 | BPE | 494 | 196 | 0.00202 | 0.00810 | 0.01619 |
| C 18K/512 | WT103 | SP | 479 | 212 | 0.00000 | 0.00209 | 0.01253 |
| A 8K/256 | WT103 | Word | 180 | 180 | 0.04444 | 0.11111 | 0.16667 |
| A 8K/256 | WT103 | BPE | 494 | 87 | 0.00405 | 0.01012 | 0.01822 |
| A 8K/256 | WT103 | SP | 452 | 121 | 0.00443 | 0.01106 | 0.02876 |
| B 8K/512 | WT103 | Word | 180 | 180 | 0.02222 | 0.09444 | 0.17222 |
| B 8K/512 | WT103 | BPE | 494 | 87 | 0.00202 | 0.00607 | 0.01012 |
| B 8K/512 | WT103 | SP | 452 | 121 | 0.00000 | 0.00443 | 0.00443 |
| C 18K/512 | WT2 | Word | 258 | 258 | 0.00775 | 0.02713 | 0.05814 |
| C 18K/512 | WT2 | BPE | 494 | 198 | 0.00000 | 0.00000 | 0.00202 |
| C 18K/512 | WT2 | SP | 463 | 145 | 0.00000 | 0.00000 | 0.00216 |
| B 8K/512 | WT2 | Word | 171 | 171 | 0.00585 | 0.05848 | 0.10526 |
| B 8K/512 | WT2 | BPE | 494 | 94 | 0.00000 | 0.00202 | 0.00202 |
| B 8K/512 | WT2 | SP | 418 | 114 | 0.00000 | 0.00239 | 0.00239 |
| A 8K/256 | WT2 | Word | 171 | 171 | 0.00000 | 0.00585 | 0.01170 |
| A 8K/256 | WT2 | BPE | 494 | 94 | 0.00000 | 0.00000 | 0.00000 |
| A 8K/256 | WT2 | SP | 418 | 114 | 0.00000 | 0.00479 | 0.00479 |

---

### 3g. WikiText Rare-Word Tokenizer-Aware Reconstruction

Reconstructs rare words from their subword pieces sequentially (piece-by-piece accuracy). Exact match requires all pieces correct.

| Config | Dataset | Tokenizer | Usable | Exact Match | Avg Piece Accuracy |
|--------|---------|-----------|---------|-------------|-------------------|
| D 18K/768 | WT103 | Word | 500 | 0.004 | 0.004000 |
| D 18K/768 | WT103 | BPE | 494 | 0.000 | 0.005061 |
| D 18K/768 | WT103 | SP | 479 | 0.000 | 0.029749 |
| C 18K/512 | WT103 | Word | 500 | 0.000 | 0.000000 |
| C 18K/512 | WT103 | BPE | 494 | 0.000 | 0.002699 |
| C 18K/512 | WT103 | SP | 479 | 0.000 | 0.038692 |
| A 8K/256 | WT103 | Word | 500 | 0.000 | 0.000000 |
| A 8K/256 | WT103 | BPE | 494 | 0.000 | 0.012045 |
| A 8K/256 | WT103 | SP | 452 | 0.000 | 0.061025 |
| B 8K/512 | WT103 | Word | 500 | 0.000 | 0.000000 |
| B 8K/512 | WT103 | BPE | 494 | 0.000 | 0.013495 |
| B 8K/512 | WT103 | SP | 452 | 0.000 | 0.044137 |
| C 18K/512 | WT2 | Word | 500 | 0.000 | 0.000000 |
| C 18K/512 | WT2 | BPE | 494 | 0.000 | 0.000000 |
| C 18K/512 | WT2 | SP | 463 | 0.000 | 0.030238 |
| B 8K/512 | WT2 | Word | 500 | 0.000 | 0.000000 |
| B 8K/512 | WT2 | BPE | 494 | 0.000 | 0.002868 |
| B 8K/512 | WT2 | SP | 418 | 0.000 | 0.042567 |
| A 8K/256 | WT2 | Word | 500 | 0.000 | 0.000000 |
| A 8K/256 | WT2 | BPE | 494 | 0.000 | 0.003374 |
| A 8K/256 | WT2 | SP | 418 | 0.000 | 0.043341 |

**Key observations:**
- SP consistently outperforms BPE on piece accuracy across all configs
- SP with 8K/256 WT103 achieves the highest piece accuracy (6.1%) — the smaller vocab means more pieces per word but each piece is more likely to appear in context
- Word tokenizer gets near-zero piece accuracy (rare words are OOV or only marginally represented)
- WT103 configs substantially improve SP piece accuracy vs WT2 (e.g., SP 18K/512: 3.9% vs WT2: 3.0%)

---

### 3h. Rare-Word Analysis by Word Length

BPE (WikiText-103, 8K/256 config, piece accuracy)

| Word Length Bucket | Count | Piece Accuracy |
|-------------------|-------|----------------|
| 3–5 chars | 72 | 0.0387 |
| 5–7 chars | 122 | 0.0227 |
| 7–9 chars | 190 | 0.0317 |
| 9–12 chars | 91 | 0.0134 |
| 12–20 chars | 25 | 0.0433 |

SP (WikiText-103, 8K/256 config, piece accuracy)

| Word Length Bucket | Count | Piece Accuracy |
|-------------------|-------|----------------|
| 3–5 chars | 66 | 0.0417 |
| 5–7 chars | 116 | 0.0292 |
| 7–9 chars | 185 | 0.0793 |
| 9–12 chars | 87 | 0.0840 |
| 12–20 chars | 23 | 0.0594 |

SP (WikiText-103, 18K/768 config, piece accuracy)

| Word Length Bucket | Count | Piece Accuracy |
|-------------------|-------|----------------|
| 3–5 chars | 65 | 0.0141 |
| 5–7 chars | 119 | 0.0266 |
| 7–9 chars | 186 | 0.0564 |
| 9–12 chars | 87 | 0.0640 |
| 12–20 chars | 24 | 0.0583 |

**Observation:** Longer words (7–12 chars) tend to have higher SP piece accuracy — they break into more recognizable subword pieces that the model can predict individually.

---

### 3i. Rare-Word Analysis by Number of Subword Pieces

BPE (WikiText-103, 18K/768 config)

| # Pieces | Count | Top-1 Accuracy |
|----------|-------|----------------|
| 1 | 1 | 0.0 |
| 2 | 127 | 0.0 |
| 3 | 234 | 0.0 |
| 4 | 106 | 0.0 |
| 5 | 25 | 0.0 |
| 6 | 7 | 0.0 |

SP (WikiText-103, 18K/768 config)

| # Pieces | Count | Top-1 Accuracy |
|----------|-------|----------------|
| 2 | 174 | 0.0 |
| 3 | 167 | 0.0 |
| 4 | 92 | 0.0 |
| 5 | 40 | 0.0 |
| 6 | 6 | 0.0 |
| 7 | 2 | 0.0 |

BPE (WikiText-103, 8K/256 config) — more pieces due to smaller vocab

| # Pieces | Count | Top-1 Accuracy |
|----------|-------|----------------|
| 2 | 73 | 0.0 |
| 3 | 189 | 0.0 |
| 4 | 157 | 0.0 |
| 5 | 65 | 0.0 |
| 6 | 16 | 0.0 |

> Confirming the smaller 8K vocab forces rare words into more subword pieces (avg ~3.5 pieces vs ~2.8 for 18K), making reconstruction proportionally harder.

---

## 4. Cross-Domain Comparison: WikiText vs Shakespeare Rare Words

Summary of how each model generalizes from its training domain to the Shakespeare rare-word test set.

| Config | Dataset | wiki_usable | wiki_top1* | wiki_top10 | wiki_top50 | wiki_top100 | shake_usable | shake_top10 | shake_top50 | shake_top100 |
|--------|---------|-------------|-----------|-----------|-----------|------------|--------------|------------|------------|-------------|
| D 18K/768 | WT103 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.000 | 0.000 | 0.000 |
| D 18K/768 | WT103 | SP: 481 | 0.0 | 0.0 | 0.0 | 0.0 | 479 | 0.000 | 0.002 | 0.010 |
| C 18K/512 | WT103 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.002 | 0.008 | 0.016 |
| C 18K/512 | WT103 | SP: 481 | 0.0 | 0.0 | 0.002 | 0.003 | 479 | 0.000 | 0.002 | 0.013 |
| A 8K/256 | WT103 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.004 | 0.010 | 0.018 |
| A 8K/256 | WT103 | SP: 477 | 0.0 | 0.0 | 0.0 | 0.0 | 452 | 0.004 | 0.011 | 0.029 |
| B 8K/512 | WT103 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.002 | 0.006 | 0.010 |
| B 8K/512 | WT103 | SP: 477 | 0.0 | 0.0 | 0.0 | 0.0 | 452 | 0.000 | 0.004 | 0.004 |
| C 18K/512 | WT2 | Word: 166 | 0.030 | 0.090 | 0.157 | 0.211 | 258 | 0.008 | 0.027 | 0.058 |
| C 18K/512 | WT2 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.000 | 0.000 | 0.002 |
| C 18K/512 | WT2 | SP: 485 | 0.0 | 0.0 | 0.002 | 0.002 | 463 | 0.000 | 0.000 | 0.002 |
| B 8K/512 | WT2 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.000 | 0.002 | 0.002 |
| B 8K/512 | WT2 | SP: 450 | 0.0 | 0.0 | 0.0 | 0.002 | 418 | 0.000 | 0.002 | 0.002 |
| A 8K/256 | WT2 | BPE: 500 | 0.0 | 0.0 | 0.0 | 0.0 | 494 | 0.000 | 0.000 | 0.000 |
| A 8K/256 | WT2 | SP: 450 | 0.0 | 0.0 | 0.0 | 0.0 | 418 | 0.000 | 0.005 | 0.005 |

*wiki_top1: directly from logs where visible; 0.0 for all WT103 configs

**Observations:**
- WT2 Config C word tokenizer shows interesting asymmetry: 21% top-100 on its home domain (WikiText rare words) but only 5.8% on Shakespeare — demonstrating domain shift
- Shakespeare rare-word accuracy is consistently higher than WikiText rare-word accuracy for WT103 configs — counterintuitive, but reflects that Shakespeare vocabulary overlaps better with the Shakespeare-derived evaluation word list
- 8K/256 WT103 achieves the best Shakespeare SP top-100 (2.9%) — the smaller vocab may learn more generalizable subword representations
- No config achieves meaningful WikiText rare-word accuracy at any top-N on WT103 (all ≤ 0.003)

---

## 5. Key Findings Summary

### 5a. Training Data Scale (WikiText-2 vs WikiText-103)

| Metric | WikiText-2 | WikiText-103 | Change |
|--------|-----------|-------------|--------|
| Word token count | ~2.1M | ~103M | +49× |
| Final word train loss (18K/512) | 4.06 | 3.24 | −20% |
| Final BPE train loss (18K/512) | 6.25 | 4.51 | −28% |
| Word emb. delta (18K/512) | +0.208 | +0.156 | −25% |
| BPE emb. delta (18K/512) | +0.352 | +0.328 | −7% |
| Word emb. mean sim. (18K/512) | 0.214 | 0.414 | +93% |
| BPE emb. mean sim. (18K/512) | 0.686 | 0.833 | +21% |
| Wiki medium-freq top-10 (word) | 0.180 | 0.0 | (OOV issue) |
| Shakespeare top-100 (word) | 0.058 | 0.118 | +103% |

The most striking effect of WT103: **absolute embedding similarity** roughly doubles for the word tokenizer, and Shakespeare rare-word accuracy more than doubles. The *delta* (above random baseline) appears to shrink because the random baseline also rises sharply with more training — the full embedding space is more tightly organized.

### 5b. Hidden Size Effect (within WikiText-103)

| Hidden | BPE emb. delta | Word emb. delta | BPE train loss | Params |
|--------|--------------|----------------|--------------|--------|
| 256 | +0.459 | +0.267 | 4.534 | 5.3M |
| 512 | +0.328 | +0.156 | 4.508 | 22.2M |
| 768 | +0.275 | +0.088 | 4.190 | 43.0M |

Counterintuitively, the larger hidden size (768) shows *lower* embedding delta than smaller configs. Two likely causes:
1. With only 3 epochs, the 768-dim model (43M params) is less converged — it needs more training time
2. The larger model's higher random-baseline similarity compresses the delta metric

### 5c. Vocabulary Size Effect (within WikiText-103)

| Vocab | Hidden | BPE emb. delta | Word emb. delta | Shakespeare word top-100 |
|-------|--------|--------------|----------------|------------------------|
| 8K | 256 | +0.459 | +0.267 | 0.167 |
| 8K | 512 | +0.431 | +0.253 | 0.172 |
| 18K | 512 | +0.328 | +0.156 | 0.118 |
| 18K | 768 | +0.275 | +0.088 | 0.103 |

Smaller vocabularies yield higher embedding deltas. This is partly because 8K vocab has a lower random-baseline similarity (smaller embedding matrices are easier to organize meaningfully). Despite lower absolute similarity, 8K models generalize better to Shakespeare rare words.

### 5d. Tokenizer Comparison

Across all configs:
- **Word tokenizer** dominates on frequent and medium-frequency word prediction (where words are in-vocabulary) but fails completely on rare words unless the vocab is large enough (18K)
- **SP consistently outperforms BPE** on piece-level rare-word reconstruction (avg piece accuracy typically 2–4× higher than BPE)
- **BPE** has the highest raw embedding similarity in larger models but SP has higher deltas in smaller models
- Neither BPE nor SP can predict rare words with any meaningful accuracy at these model scales — this appears to be a fundamental limitation requiring either much more data or much larger models

---

## 6. Limitations and Caveats

1. **Truncated pandas output:** The benchmark tables in the raw logs had columns `top1` and `top5` truncated by default pandas display settings. These metrics were computed but not directly captured for all runs. Where explicitly visible in logs they are included; otherwise noted.

2. **WT103 word vocabulary:** The word-level tokenizer vocabulary is built from Shakespeare training text, not WikiText-103. This explains why word tokenizer gets 0 usable examples on WikiText medium/rare word benchmarks for all WT103 configs — the evaluation words are simply OOV.

3. **3 epochs on WT103:** While token-count equivalent to ~150 WT2 epochs, 3 training passes may not be optimal for the 768-dim model (43M params). A fair comparison would require equalizing either epochs or wall-clock time.

4. **Small evaluation sets:** The frequent-word benchmark on WT103 has only 1 usable example — results are not statistically meaningful and are presented for completeness only.

5. **No exact rare-word recovery:** No config achieves even 0.5% exact-match rare-word reconstruction. This is expected — these are small models (5–43M params) trained for hours, not days, on a single GPU.

---

## 7. Log File Reference

| Log File | Config | Dataset | SLURM Job ID |
|----------|--------|---------|-------------|
| `abl_18k_768_103_15375.out` | D 18K/768 | WikiText-103 | 15375 |
| `abl_18k_512_103_15372.out` | C 18K/512 | WikiText-103 | 15372 |
| `abl_8k_256_103_15373.out` | A 8K/256 | WikiText-103 | 15373 |
| `v3_8k_512_103_15374.out` | B 8K/512 | WikiText-103 | 15374 |
| `ablation_18k_512_15363.out` | C 18K/512 | WikiText-2 | 15363 |
| `ablation_8k_256_15364.out` | A 8K/256 | WikiText-2 | 15364 |
| `v3_8k_512_15365.out` | B 8K/512 | WikiText-2 | 15365 |

All logs are in `/aiau010_scratch/tzs0128/logs/` and a copy is committed to the repository under `logs/`.
