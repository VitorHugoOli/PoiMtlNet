# gradient_cosine_tests.csv -- read the `unit` column before quoting any p-value

Each dataset appears at every level of aggregation, because the level decides what the p-value means.

`observation (epoch-fold, SERIALLY DEPENDENT)` -- one row per (fold, epoch). These 50 values per fold
are consecutive epochs of ONE training trajectory, so they are NOT independent draws. A t-test, TOST
or sign test computed here treats n=250 or n=3150 as i.i.d. and is ANTI-CONSERVATIVE: the p-value is
smaller than the evidence supports. Reported here only for continuity with the raw data; do NOT quote
its p-values as significance.

`fold-series mean` -- one value per fold. A fold is one train/validation split, so this is the
independent unit of the protocol. n=5 per dataset (n=60 for Florida, five folds x twelve
configurations). THIS IS THE UNIT TO QUOTE.

`configuration mean` -- Florida only. Its twelve configurations reuse the same five folds, so the
configurations are not independent of one another either; averaging to one value per configuration
(n=12) is the most conservative reading available.

`sign_p_floor_at_this_n` -- the smallest p the sign test could return at that n if EVERY value agreed
in direction. At n=5 the floor is 0.0625, so a distribution-free test cannot reach 0.05 no matter how
consistent the effect. Any "significant" claim at n=5 therefore rests entirely on the normality
assumption of the t-test, and should be reported as suggestive rather than established.

## What survives aggregation, and what does not

ROBUST: equivalence to zero within +/-0.05 holds by TOST at every dataset and every unit. This is the
appendix's claim and it does not depend on the dependence question.

NOT ROBUST: Alabama's positive mean. At the fold level its five fold means are all positive
(mean +0.0112, sd 0.0058) and the t-test rejects (p=0.0125), but the sign test cannot (p=0.0625, which
is its floor at n=5). Report it as a consistent positive tendency on five folds, not as a significant
effect.
