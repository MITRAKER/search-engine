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
| BM25F, title x3 | scratch | 16.81 | 15.31 | 34.63 | 42.67 |  |
| BM25F, equal weights | scratch | 17.79 | 16.14 | 38.93 | 44.91 |  |
| TF-IDF, title x3 | scratch | 6.58 | 5.91 | 14.32 | 16.92 |  |
| combined_fields, title x3 | es | 8.26 | 7.68 | 11.57 | 17.92 | 5.00 |
| best_fields, tie_breaker 0.5 | es | 7.29 | 7.01 | 9.77 | 14.73 | 5.00 |

Time to build the index: scratch 4.78 s, es 3.29 s.
