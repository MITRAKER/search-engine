**SciFact (dev claims)**: 5,183 documents, 300 judged queries.

| System | Engine | nDCG@10 | MRR@10 | P@10 | Recall@100 | MAP@100 | p (nDCG vs row 1) |
|---|---|---|---|---|---|---|---|
| BM25F, title x3 | scratch | 0.6860 | 0.6505 | 0.0900 | 0.9276 | 0.6450 |  |
| BM25F, equal weights | scratch | 0.6798 | 0.6442 | 0.0893 | 0.9276 | 0.6393 | 0.1305 |
| TF-IDF, title x3 | scratch | 0.6524 | 0.6138 | 0.0873 | 0.9092 | 0.6071 | 0.0003 |
| combined_fields, title x3 | es | 0.6860 | 0.6505 | 0.0900 | 0.9276 | 0.6450 | 1.0000 |
| best_fields, tie_breaker 0.5 | es | 0.6904 | 0.6564 | 0.0907 | 0.9142 | 0.6501 | 0.5130 |

| System | Engine | Mean ms | Median ms | p95 ms | p99 ms | Inside Elasticsearch, median ms |
|---|---|---|---|---|---|---|
| BM25F, title x3 | scratch | 17.42 | 15.72 | 36.61 | 42.87 |  |
| BM25F, equal weights | scratch | 18.70 | 17.09 | 42.40 | 50.34 |  |
| TF-IDF, title x3 | scratch | 5.75 | 5.28 | 11.64 | 14.22 |  |
| combined_fields, title x3 | es | 6.84 | 6.62 | 8.43 | 12.11 | 4.00 |
| best_fields, tie_breaker 0.5 | es | 6.31 | 6.05 | 8.50 | 10.42 | 4.00 |

Time to build the index: scratch 4.49 s, es 2.98 s.
