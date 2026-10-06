# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: one deal, 3,022 companies, 139 same-as edges, 42
documents, 33 risks with their evidence, 40 applications, 240 comparable deals. Reproduce with `make demo`, then `make load-test`.

## Reads: the risk register with evidence and the deal summary, alternating, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1060 | 106.0 req/s | 9.4 ms | 16.6 ms | 21.8 ms | 0.00% |
| 8 | 1964 | 196.1 req/s | 38.0 ms | 55.8 ms | 66.9 ms | 0.00% |
| 32 | 1531 | 150.5 req/s | 172.1 ms | 214.8 ms | 230.2 ms | 0.00% |

The risk register joins every risk to its evidence and documents and, for a synthetic deal, scores the parsers and the matcher against
the generator (the data room is cached per process). A first run of this test, taken while several other projects were building Docker
images on the same machine, measured 3.9 req/s at concurrency 1; the table above is a rerun on a quieter machine.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the data room (financials, 3,020 customers, 41 documents, 40 applications) and 240 past deals | 0.12 s |
| Match 420 customers against 2,600 | 0.01 s |
| Parse 41 documents and cluster 40 applications | under 0.01 s |
| Fit the synergy model (two maximum-likelihood fits over 417 and 326 reports) and the ERP model | 0.45–0.65 s |
| Integration scenario: team size by enumeration (14 options x 4,000 draws), then 10,000 draws per option and the truth | 0.4 s |
| The whole demo scenario against a test database | 1.7 s |
| Evaluation: five data rooms, 2,000 fresh deals and 300 hidden fates each | 12 s |

Not measured: a real data room of thousands of documents (parsing is linear and cheap; matching would need blocking beyond the first
word at a million customers), or many deals per tenant.
