# Evaluation

Five data rooms and deal databases the demo never uses (seeds 41, 51, 61, 71, 81). Each model is fitted the way the API
fits it and scored against what the generator knows. Reproduce with `python -m ma.evaluate`.

## Synergy realisation, on 2,000 fresh deals per database

Realised run-rate as a share of the announced run-rate three years after close. Error is the median absolute error in that share (0.10 = ten points).

| Database | Kind | Model | Management case (1.0) | Bankers' comparables | Latest disclosure as is | Model's 80% interval covers | Model median vs truth |
|---|---|---|---|---|---|---|---|
| 41 | cost | 0.134 | 0.182 | 0.168 (level 0.94) | 0.174 | 75% | +1% |
| 41 | revenue | 0.160 | 0.512 | 0.219 (level 0.60) | 0.171 | 80% | -4% |
| 51 | cost | 0.133 | 0.179 | 0.167 (level 0.97) | 0.169 | 79% | +0% |
| 51 | revenue | 0.157 | 0.525 | 0.231 (level 0.63) | 0.167 | 77% | +5% |
| 61 | cost | 0.138 | 0.185 | 0.184 (level 1.00) | 0.181 | 83% | +2% |
| 61 | revenue | 0.155 | 0.514 | 0.275 (level 0.71) | 0.178 | 81% | -4% |
| 71 | cost | 0.137 | 0.177 | 0.178 (level 1.00) | 0.172 | 81% | +2% |
| 71 | revenue | 0.162 | 0.523 | 0.272 (level 0.69) | 0.165 | 81% | +1% |
| 81 | cost | 0.141 | 0.186 | 0.193 (level 1.01) | 0.189 | 78% | +3% |
| 81 | revenue | 0.161 | 0.528 | 0.263 (level 0.67) | 0.174 | 79% | -2% |

The comparables are the deals at least three years old that still disclose. Deals that fell behind tend to stop disclosing, so the
comparables run high; taking every latest disclosure as final runs low, because a third of the deals are still ramping.

## ERP programmes

| Database | Programmes (running) | Lateness, censored fit | Finished only | Truth | Incident rate fast / staged | Truth |
|---|---|---|---|---|---|---|
| 41 | 149 (17) | ×1.33 | ×1.33 | ×1.35 | 29% / 9% | 35% / 8% |
| 51 | 135 (18) | ×1.31 | ×1.30 | ×1.35 | 47% / 7% | 35% / 8% |
| 61 | 138 (11) | ×1.31 | ×1.30 | ×1.35 | 41% / 7% | 35% / 8% |
| 71 | 149 (21) | ×1.40 | ×1.38 | ×1.35 | 42% / 7% | 35% / 8% |
| 81 | 156 (26) | ×1.32 | ×1.29 | ×1.35 | 44% / 6% | 35% / 8% |

Only 8–17% of the programmes are still running in these databases, so dropping them moves the estimate by at most 0.03 here; the
censored fit matters more when a database is young. Lateness is recovered to within 0.05; the fast incident rate, from 40–60 finished
fast programmes, is off by up to twelve points.

## Fast against staged ERP, for the demo deal's terms on each data room

| Data room | Fast: model / truth / plan as written | Staged: model / truth / plan as written | Fast minus staged: model / truth | Same choice | 80% range, fast (model) | P(covers the $260M premium), fast: model / truth |
|---|---|---|---|---|---|---|
| 41 | $300M / $303M / $396M | $297M / $311M / $388M | +3 / -8 | no | $186M | 68% / 67% |
| 51 | $282M / $295M / $396M | $281M / $300M / $388M | +1 / -5 | no | $181M | 58% / 64% |
| 61 | $291M / $286M / $396M | $289M / $286M / $387M | +2 / -0 | no | $201M | 61% / 60% |
| 71 | $284M / $300M / $396M | $300M / $306M / $387M | -17 / -6 | yes | $183M | 59% / 66% |
| 81 | $299M / $304M / $397M | $306M / $312M / $389M | -7 / -8 | yes | $187M | 68% / 68% |

The model picks the truth's better option on 2 of 5 data rooms: no better than a coin. The truth's own gap between the
options is at most $8M against an 80% range of $180–200M, so a wrong pick costs little; the question the model does answer is how
far the plan as written ($77–110M above the truth) sits from either option, and whether the premium is covered.

## Tracking realisation: 300 hidden fates per data room, four quarterly reports each

| Data room | 80% interval covers, cost / revenue | Median error, cost: tracker / straight line / no reports | Revenue: tracker / straight line / no reports |
|---|---|---|---|
| 41 | 78% / 88% | 0.026 / 0.233 / 0.149 | 0.013 / 0.232 / 0.168 |
| 51 | 81% / 79% | 0.025 / 0.223 / 0.131 | 0.012 / 0.190 / 0.128 |
| 61 | 68% / 76% | 0.035 / 0.206 / 0.128 | 0.010 / 0.165 / 0.121 |
| 71 | 49% / 74% | 0.046 / 0.222 / 0.125 | 0.013 / 0.218 / 0.153 |
| 81 | 78% / 78% | 0.023 / 0.205 / 0.138 | 0.016 / 0.205 / 0.155 |

The straight line extends the latest report to month 36 in proportion. The tracker divides by a ramp fitted to the deal database, which has
the same shape as the generator's (a power of time); a real ramp of another shape would hurt it. Its cost intervals are too narrow on
two databases (68% and 49% coverage): the fitted cost ramp exponent comes out 0.81–0.83 against a true 0.80, three or four of its own
standard errors high, because deals that go quiet are not missing at random once revenue figures are missing too, and the model is
linear where the truth has kinks. A few hundredths on the exponent moves a month-3 report by about 5%.

## Customer matching, contract rules, application clusters

| Data room | Matcher precision / recall | Exact name precision / recall | Overlap share: matcher / exact / truth | Change of control found / false / present | Convenience found / false / present | Clusters precision / recall | Same product precision / recall |
|---|---|---|---|---|---|---|---|
| 41 | 0.90 / 1.00 | 1.00 / 0.30 | 23% / 6% / 21% | 12 / 0 / 15 | 5 / 0 / 5 | 1.00 / 1.00 | 1.00 / 0.28 |
| 51 | 0.91 / 1.00 | 1.00 / 0.25 | 39% / 8% / 29% | 15 / 0 / 19 | 8 / 0 / 8 | 1.00 / 0.80 | 1.00 / 0.47 |
| 61 | 0.88 / 1.00 | 1.00 / 0.23 | 42% / 5% / 40% | 10 / 0 / 15 | 9 / 0 / 9 | 1.00 / 0.95 | 1.00 / 0.58 |
| 71 | 0.86 / 1.00 | 1.00 / 0.18 | 30% / 6% / 24% | 12 / 0 / 13 | 9 / 0 / 9 | 1.00 / 1.00 | 1.00 / 0.38 |
| 81 | 0.89 / 1.00 | 1.00 / 0.22 | 24% / 6% / 22% | 10 / 0 / 12 | 12 / 0 / 12 | 1.00 / 0.86 | 1.00 / 0.43 |

Change-of-control recall by wording, all five data rooms:

- "In the event of a Change of Control of Supplier, Customer may terminate this Agreement upon thirty (30) days' written notice.": 15 of 15
- "Customer may terminate this Agreement if Supplier undergoes a change in control, merger or sale of substantially all of its assets.": 13 of 13
- "This Agreement may not be assigned, including by operation of law or by merger, without Customer's prior written consent.": 15 of 15
- "Should Supplier be acquired by, or merged into, any third party, Customer shall have the right to end this Agreement without penalty.": 16 of 16
- "If more than fifty percent (50%) of the voting securities of Supplier are transferred, Customer may elect to terminate.": 0 of 15

These three are circular. The matcher's normalisation, the contract rules and the tags were written by the same hand as the generator's
spelling variants, clause wordings and tag sets, so recall here is an upper bound. The matcher's lost precision is the sister entities
(the same name with another legal suffix), which a normaliser that drops suffixes cannot tell apart. The rules miss the one wording
that never names a change of control.

Run time 10 s on one core.
