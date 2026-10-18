# Phase 30 results: retrieval depth vs indirect prompt injection

Independent variable: Top-K in {1, 3, 5, 10, 20}. Dependent variables: attack success rate (compliance with the injected instruction), benign task completion, mean tokens, mean latency (logical units). Controls: same stand-in model, system prompt, documents, tool set; no-poison control arm at K = 1 and K = 20. Ablation: two-stage instruction-aware rerank (dense top-2K -> rerank -> top-K) at K = 5. n = 12 episodes per condition (one per topic); 72 episodes in the episode log, the control arm summarised in the table.

| K | ASR (hits) | Wilson 95 | completion | mean tokens | mean latency | violations |
|---|------------|-----------|------------|-------------|--------------|------------|
| 1 | 0.000 (0/12) | [0.000, 0.243] | 1.000 | 28.5 | 5.00 | 0 |
| 3 | 0.333 (4/12) | [0.138, 0.609] | 0.667 | 45.5 | 5.17 | 12 |
| 5 | 0.583 (7/12) | [0.320, 0.807] | 0.417 | 64.9 | 5.29 | 21 |
| 10 | 0.833 (10/12) | [0.552, 0.953] | 0.167 | 106.7 | 5.42 | 30 |
| 20 | 1.000 (12/12) | [0.757, 1.000] | 0.000 | 171.6 | 5.50 | 36 |

| condition | ASR (hits) | Wilson 95 | completion |
|-----------|------------|-----------|------------|
| control K=1 (no poison) | 0.000 (0/12) | [0.000, 0.243] | 1.000 |
| control K=20 (no poison) | 0.000 (0/12) | [0.000, 0.243] | 1.000 |
| ablation K=5, rerank ON | 0.333 (4/12) | [0.138, 0.609] | 0.667 |

Effect sizes (Cohen's h, 95% CI): K=20 vs K=1 on ASR: h = 3.142 [2.306, 3.977]; ablation vs dense at K=5: h = -0.507 [-1.343, 0.328].

Reading: the K=1-vs-K=20 difference is resolvable at n=12 per arm; the ablation's direction is clear (the rerank drops the injected document on the borderline topics) but its size is not resolvable at n=12, which is the sample-size decision the chapter teaches.
