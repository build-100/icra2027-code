# Release validation

Validation was performed in the independent Python 3.12 environment documented
in ENVIRONMENT.md. CPU was used for the software checks.

- 46 scientific regression tests passed after package migration. Four tests use
  synthetic seven-node routes in place of private manuscript result fixtures,
  retaining their success, reward-decomposition and auxiliary-credit assertions.
- 89 local saved routes replayed with matching scores and success, requiring
  1,150 read-only simulator steps. All nine calibration arithmetic checks passed.
- A 600-call GEVD smoke run produced 199 episodes and 228 optimizer updates.
  Selected and current-policy results, counts and updates matched the previous
  implementation's corresponding QA run. This is not a manuscript experiment.
- QMIX, COMA and QCO short-run logs matched their previous QA records;
  dGre routes, scores and simulator counts also matched.
- All seven baseline entries and four ablation entries passed short checks.
  QMIX at 300 calls made 24 optimizer updates; QCO and QCF made 22 updates each.
- Historical GEVD and baseline checkpoints were loaded through restricted
  compatibility mappings, without importing the removed source tree.
- The source/configuration/figure manifest can be checked with
  `python scripts/verify_release.py`. Private route checks are opt-in through
  `--local-results`; public-clone checks do not require unpublished data.

No formal batch was resumed, no seed budget was extended, and no physical robot
experiment was conducted during this refactoring.
