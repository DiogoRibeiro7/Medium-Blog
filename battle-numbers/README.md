# Numbers and victory in 660 battles, 1600 to 1982

Companion code for the Medium article on what CDB90, the US Army Concepts Analysis Agency's database of land battles, says about Lanchester's laws: how much numbers decide a battle, and what else does.

## What it finds

All numbers below are from `output.txt`.

**Numbers decide less than Lanchester's square law needs.**
- The larger side won 60% of the decisive battles with unequal numbers (95% interval 55% to 65%).
- Doubling the force ratio raises the attacker's odds by a factor of 1.39, or 1.61 when battles are compared only within the same war.
- For the square law to fit, one side would typically have to be about 200 to 2,000 times as effective as the other.

**The losses run the wrong way for the square law.**
- The square law predicts that the attacker's share of the losses falls in proportion to the force ratio: slope −1.
- The data give +0.29 (+0.48 holding the result fixed).
- Stated errors in the strengths shrink these slopes by only about 6%.

**Battles end long before anyone is destroyed.** The losing side's median loss was 16% of its initial strength, and the winner's 6%. The losing side lost under a tenth of its strength in 36% of decisive battles and over half in 12%.

**Exceptions both ways.** Attackers with under half the defender's numbers won 13 battles, 5 of them with surprise; attackers with over three times the defender's numbers lost 25. Both lists are printed in `output.txt`.

**What else goes with winning.** Each measure is known or countable before the battle:
- **Surprise.** The attacker won 79% of battles in which he achieved surprise, against 61% of battles with no surprise either way.
- **The commanders' earlier records.** The attacker won 74% when his commander had the better record in earlier CDB90 battles, against 35% when the record was worse.
- **Terrain.** Coded terrain shows no association with the result.
- **Fortified defences.** These are met with larger attacking forces (median 2.1:1, against 1.1:1 for hasty defences), so their effect is hidden by the commanders' own choices.

**Coded advantages explain most of the outcome, but the coders knew it.**
- **How much they explain.** HERO's ten coded advantages (leadership, morale and others) explain 77% of the variation in who won. Numbers alone explain 4%; everything measurable before the battle explains 25%.
- **Same opponent, different code.** Where the same two commanders met again with a different result, the coded leadership advantage moved with the result (within-pair correlation 0.37). Against Archduke Charles in 1809, Napoleon is coded +1 at Eckmühl and Wagram, which he won, and 0 at Aspern-Essling, which he lost.

**By era.** The larger side won 53% of battles in 1600–1799 and in 1800–1899, 69% in 1900–1945, and 47% in 1946–1982.

## Data

CDB90 (CAA Database of Battles, Version 1990), in the tidy open-data version by Jeffrey B. Arnold:
- repository [jrnold/CDB90](https://github.com/jrnold/CDB90), commit `8d984f1`;
- archived on Zenodo, doi:10.5281/zenodo.23928303;
- licence CC0 1.0.

The four tables used here are committed in `data/`. `fetch_data.py` downloads them from that commit and checks each against the SHA-256 digests in `digests.txt`.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on synthetic battles with a known answer
python battle_numbers.py > output.txt      # every number in the article
python figures.py                          # figures/
python fetch_data.py --check               # data/ matches the pinned source
```

The workflow `.github/workflows/battle-numbers.yml` runs all of this and fails if `output.txt` changes.

## Files

- **`battle_core.py`**: force ratios, win and loss models, Lanchester benchmarks, prior records, the within-commander and within-pair hindsight checks, and the bootstrap. Intervals resample whole campaigns; a within-war conditional logit is the check at the level of wars.
- **`battle_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the five figures.
- **`tests/test_core.py`**: square-law and linear-law losses give slopes of −1 and 0; the implied quality spread is recovered from simulated square-law battles; prior records count only earlier battles; a code read off the result is caught by the permutation test and an unrelated code is not.

## Limits

- **A selected record.** CDB90 is a selection: the battles HERO studied, mostly Western, many from the Second World War. Battles happen when both sides accept them, so it is a record of chosen fights, not a random sample of possible ones.
- **Coded victory.** Victory itself is a coded judgement.
- **Earlier records.** Prior records cover only battles in CDB90. They mix the commander with the army he led.

## Licence

Code: Apache-2.0. Data: CC0 1.0, as above.
