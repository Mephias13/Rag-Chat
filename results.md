# Results

83 questions, 100 articles, 1731 chunks. Embeddings: bge-m3. LLM router: qwen3.5:4b (zero-shot). Intervals are 95% bootstrap (2000 resamples of questions, seed 0).


## E1. Retrieval, whole knowledge base (article level)

| method | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|
| bm25_raw | 0.59 [0.48, 0.69] | 0.85 [0.77, 0.93] | 0.70 [0.61, 0.78] |
| bm25_stem | 0.68 [0.57, 0.77] | 0.92 [0.85, 0.96] | 0.76 [0.67, 0.83] |
| dense | 0.80 [0.70, 0.88] | 0.95 [0.90, 0.99] | 0.86 [0.80, 0.92] |
| hybrid | 0.77 [0.68, 0.85] | 0.95 [0.90, 0.99] | 0.85 [0.78, 0.91] |

Paired differences (bootstrap over the same resampled questions):

| pair | d hit@5 | d MRR@10 |
|---|---|---|
| hybrid - dense | 0.00 [-0.04, 0.04] | -0.01 [-0.06, 0.03] |
| hybrid - bm25_stem | 0.04 [0.00, 0.08] | 0.09 [0.05, 0.13] |
| bm25_stem - bm25_raw | 0.06 [0.00, 0.13] | 0.06 [-0.01, 0.12] |

## E2. Topic router

| router | accuracy | macro-F1 | latency p50 / p95, ms |
|---|---|---|---|
| LLM | 0.64 [0.53, 0.75] | 0.62 [0.51, 0.72] | 2480.5 / 2563.4 |
| TF-IDF | 0.52 [0.41, 0.61] | 0.45 [0.33, 0.55] | 1.5 / 2.6 |

LLM, rows = true topic, columns = predicted:

| true \ pred | refunds | credits | cards | payments | fraud | unknown |
|---|---|---|---|---|---|---|
| refunds | 10 | 4 | 1 | 1 | 1 | 0 |
| credits | 1 | 15 | 0 | 1 | 2 | 0 |
| cards | 0 | 1 | 12 | 1 | 4 | 0 |
| payments | 0 | 3 | 6 | 4 | 3 | 0 |
| fraud | 0 | 0 | 0 | 1 | 12 | 0 |

TF-IDF, rows = true topic, columns = predicted:

| true \ pred | refunds | credits | cards | payments | fraud | unknown |
|---|---|---|---|---|---|---|
| refunds | 9 | 6 | 1 | 1 | 0 | 0 |
| credits | 2 | 16 | 1 | 0 | 0 | 0 |
| cards | 2 | 0 | 14 | 2 | 0 | 0 |
| payments | 2 | 4 | 8 | 2 | 0 | 0 |
| fraud | 1 | 5 | 5 | 0 | 2 | 0 |

## E3. Flat search vs routed search as the knowledge base grows (hybrid retrieval)

| articles in KB | search | questions | hit@5 | MRR@10 | d hit@5 vs flat |
|---|---|---|---|---|---|
| 40 | flat | 34 | 0.97 [0.91, 1.00] | 0.92 [0.85, 0.98] | - |
| 40 | router LLM | 34 | 0.56 [0.38, 0.71] | 0.54 [0.36, 0.71] | -0.41 [-0.59, -0.27] |
| 40 | router TF-IDF | 34 | 0.50 [0.32, 0.68] | 0.48 [0.32, 0.65] | -0.47 [-0.65, -0.29] |
| 40 | oracle topic | 34 | 1.00 [1.00, 1.00] | 0.95 [0.89, 1.00] | 0.03 [0.00, 0.09] |
| 70 | flat | 58 | 0.98 [0.95, 1.00] | 0.89 [0.82, 0.95] | - |
| 70 | router LLM | 58 | 0.64 [0.52, 0.76] | 0.61 [0.49, 0.74] | -0.34 [-0.47, -0.22] |
| 70 | router TF-IDF | 58 | 0.55 [0.43, 0.67] | 0.51 [0.39, 0.64] | -0.43 [-0.55, -0.31] |
| 70 | oracle topic | 58 | 1.00 [1.00, 1.00] | 0.93 [0.88, 0.97] | 0.02 [0.00, 0.05] |
| 100 | flat | 83 | 0.95 [0.90, 0.99] | 0.85 [0.78, 0.91] | - |
| 100 | router LLM | 83 | 0.64 [0.53, 0.75] | 0.59 [0.48, 0.69] | -0.31 [-0.42, -0.20] |
| 100 | router TF-IDF | 83 | 0.52 [0.41, 0.61] | 0.48 [0.38, 0.58] | -0.43 [-0.54, -0.33] |
| 100 | oracle topic | 83 | 0.99 [0.96, 1.00] | 0.91 [0.86, 0.95] | 0.04 [0.00, 0.07] |
