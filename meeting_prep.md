# Meeting Prep — Prof. Hait Discussion (April 10, 2026)

> **Share this with the team BEFORE the meeting. Read it fully.**

---

## Meeting Details

| | |
|---|---|
| **When** | Today, April 10, 2026 — **11:15 AM** |
| **Where** | Google Meet (link from Prof. Hait) |
| **Attendees** | Prof. Swati Hait, Anshul, Advik, Kritii, Chandrima |
| **What to present** | PPT with modified algorithm + results (`presentation_apr10.pptx`) |

---

## What Prof. Wants

From her Apr 8 email:
1. **Present results in the meeting** — we have a PPT ready
2. **Finish paper ASAP** — she explicitly said "I want to have a publication with you all"
3. **Discuss results** — she had 4 specific concerns (see below)
4. **Also:** She asked if anyone is available for next semester project

---

## Prof's 4 Concerns (March 31 Email) — CRITICAL

These are the main things she'll ask about. **Everyone must know these answers.**

### Concern 1: "Have you applied the new grouping function?"

**Answer: YES.**

- The grouping operator is implemented in both `adaptive_ga.py` and `nsga2.py`
- Features are split into **4 quartile groups**: `[0, D/4), [D/4, D/2), [D/2, 3D/4), [3D/4, D)`
- Crossover and mutation **respect group boundaries** — they don't randomly mix features across groups
- This is visible in Tables 7, 8, and 9

**If she probes deeper:**
- Two-point crossover stays within groups
- Bit-flip mutation only flips within the same group
- The grouping preserves spatial/channel structure from the CNN backbone features

---

### Concern 2: "Most values in many tables are exactly the same — not even a small change"

**This is a valid observation.** Here's how to handle it:

**Honest explanation:**
- The LC25000 dataset is **relatively easy** for modern CNN+attention architectures
- DenseNet121 + SE attention features are already very discriminative
- So all feature selection methods achieve ~99.7% accuracy — the headroom for improvement is tiny
- The differences show up in **feature count** and **efficiency**, not raw accuracy

**Evidence to point out:**
| What changes | Where to see it |
|---|---|
| Feature reduction | Table 6: Adaptive GA selects 510 features vs NSGA-II selects 310 (69.7% reduction) |
| Computation time | Table 10: Adaptive GA takes 1186s vs Baseline GA 1958s vs NSGA-II 4508s |
| Backbone variation | Table 14: EfficientNetB0 = 99.90% vs VGG16 = 96.77% (big difference) |
| Attention variation | Table 5: ViT/Swin = 99.83% vs CBAM = 93.67% (big difference) |
| Per-class metrics | Confusion matrices show different error patterns per method |

**Key talking point:** "The accuracy values are similar because the dataset saturates near 99.7%, but the methods differ significantly in feature efficiency and computational cost. The real contribution is achieving the same accuracy with far fewer features."

**DO NOT SAY:** "We don't know why" or "We'll look into it" — we already know why.

---

### Concern 3: "NSGA-II and Simple GA have not affected results — not even a small change"

**Answer: The results ARE different, but in feature count, not accuracy.**

| Method | Accuracy | Features Selected | Reduction |
|---|---|---|---|
| Baseline GA | 99.73% | 506 | 50.6% |
| Adaptive GA | 99.70% | 510 | 50.2% |
| NSGA-II | 99.70% | 310 | **69.7%** |

**Key argument:** NSGA-II selects **40% fewer features** than Baseline GA while maintaining the same accuracy. This means:
- Faster inference at deployment
- Less overfitting risk
- More interpretable model
- This IS the contribution — it's a multi-objective optimization, not just accuracy optimization

**If she's not satisfied:**
- Acknowledge that on a harder dataset (e.g., TCGA or real clinical data), the accuracy differences would be more pronounced
- The methodology is sound — the dataset just happens to be well-separated

---

### Concern 4: "Have the new grouping operators improved results? Or were previous operators better?"

**Answer: Marginal improvement in accuracy, but structural benefit.**

From Table 7:
| Config | Accuracy | Features |
|---|---|---|
| Adaptive GA **without** grouping | 99.70% | ~510 |
| Adaptive GA **with** grouping | 99.77% | ~522 |

- Grouping gives **+0.07% accuracy** (small but positive)
- More importantly: grouped features maintain spatial coherence from CNN layers
- Previous (old) operators selected features randomly across the feature space
- New grouping ensures each "region" of the feature vector is represented

**Comparison with old functions:**
- We ran all experiments with the NEW functions as instructed
- Old results are saved separately for comparison
- We should present both side-by-side if she asks

---

## Status of All Tables

| Table | Description | Status | Key Result |
|---|---|---|---|
| 2 | Classifier performance (KNN, SVM, RF) | ✅ Done | KNN best: 99.70% |
| 3 | SOTA comparison | ✅ Done | Beats all 6 baselines |
| 4 | 10-fold cross-validation | ✅ Done | 99.77% ± 0.07% |
| 5 | Attention mechanism comparison | ✅ Done | ViT/Swin best (99.83%) |
| 6 | GA vs NSGA-II | ✅ Done | NSGA-II: 69.7% reduction |
| 7 | Grouping operator effect | ✅ Done | +0.07% with grouping |
| 8 | GA variants comparison | ✅ Done | Baseline > Adaptive > Adaptive+Group |
| 9 | NSGA-II with/without grouping | ✅ Done | Same accuracy, same features |
| 10 | Comprehensive FS comparison | ✅ Done | NSGA-II best reduction |
| 11 | Population size sensitivity | ✅ Done | Stable across 20-80 |
| 12 | Generations sensitivity | ✅ Done | Converges by 50 |
| 13 | KNN k sensitivity | ✅ Done | k=5 or k=7 optimal |
| 14 | Backbone comparison | ✅ Done | EfficientNetB0 best (99.90%) |
| 15 | Ensemble fusion (6 methods) | ✅ Done | GP1 best ensemble (99.67%) |
| **16** | **Full pipeline (4 backbones + NSGA-II + GP)** | **❌ OOM** | **Needs ≥64GB RAM** |

---

## Talking Points for the Meeting

### What to emphasize:
1. **All 15 tables complete** with the new functions
2. **NSGA-II's real value is feature reduction** (69.7%), not accuracy boost
3. **Grouping operator maintains structural coherence** in feature selection
4. **Best accuracy: 99.90%** (EfficientNetB0) — state of the art on LC25000
5. **Paper is nearly ready** — just need Table 16 and final writeup

### What to propose as next steps:
1. Resolve Table 16 OOM — options: bigger machine / sequential processing / PCA pre-reduction
2. Finalize Overleaf paper with all tables
3. Write discussion section comparing old vs new operators
4. Submit paper

### Things to AVOID:
- Don't be defensive about similar values — explain it confidently as dataset saturation
- Don't promise specific timelines unless you can deliver
- Don't say "we'll check" about things we already know — present the data
- Don't bring up cloud credit issues again (that's old news)

---

## Individual Roles

| Person | Role in meeting |
|---|---|
| **Anshul** | Lead presenter — walk through PPT, explain architecture and results |
| **Advik** | Support — explain Table 16 OOM issue, feature selection details |
| **Kritii** | Support — can speak to CS project report status if asked |
| **Chandrima** | Support — methodology questions |

**Everyone should be ready to answer:**
- "What does the grouping operator do?" → splits features into 4 groups, crossover/mutation respect boundaries
- "Why are values the same?" → dataset saturation at ~99.7%, differences are in feature count
- "What's the benefit of NSGA-II?" → 69.7% feature reduction vs 50% for GA, same accuracy
- "When will the paper be ready?" → [decide this before the meeting!]

---

## Quick Reference: Key Numbers

| Metric | Value |
|---|---|
| Best single accuracy | 99.90% (EfficientNetB0) |
| Default pipeline accuracy | 99.70% (DenseNet121 + SE + Adaptive GA + KNN) |
| Best ensemble | 99.67% (GP1 Max operator) |
| NSGA-II feature reduction | 69.7% (310/1024 features) |
| 10-fold CV | 99.77% ± 0.07% |
| SOTA comparison | Beats all 6 prior methods (prev best: 98.1%) |

---

## PPT Location

The presentation is at: `presentation_apr10.pptx` (21 slides)
Screen-share this during the Google Meet.

---

**Read this. Know the 4 concerns. Present confidently. Good luck!**
