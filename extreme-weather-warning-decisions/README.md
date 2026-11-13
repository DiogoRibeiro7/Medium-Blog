# Extreme weather warnings: decisions under uncertainty

Public reproducibility companion for Diogo Ribeiro's article *Cost–Loss Thresholds and the Value of Waiting for the Next Forecast: Portugal's November 2026 Storm Alert*.

The manuscript is kept separately in the private [`research-articles-medium`](https://github.com/DiogoRibeiro7/research-articles-medium) repository. This folder holds the code, the printed results and the figures.

## Run

Requirements: Python 3.11 or newer, NumPy and Matplotlib.

```bash
python -m pip install -r requirements.txt
python warning_decisions.py > results/output.txt
```

The script is deterministic and downloads nothing. It prints every number quoted in the article to `results/output.txt`, writes five figures to `figures/`, and renders the nine displayed equations to `figures/equations/` (Medium does not render LaTeX, so the article shows them as images). The CI workflow reruns it and fails if the printed numbers change.

## What the model is

A two-action cost–loss problem with a later binary forecast signal. Protection costs `C`, the severe event costs `L0` without protection and `L1` with it, and the action threshold is `p* = C / (L0 − L1)`. The signal has equal sensitivity and specificity `q`. Each hour of delay raises the protected loss by a fixed amount, standing in for people already on the road and less time to act. After the signal arrives, each branch takes the cheaper of protecting and not protecting.

The script reports three things:

- the break-even delay: the longest wait for which the next forecast still beats deciding now;
- the minimum useful accuracy: below it, no reading of the signal can change the decision, so the next forecast is worth nothing even if it arrives instantly;
- the break-even recoverable share for "protect now, review when the signal arrives": the share of the protection cost that must still be avoidable for lifting the measures after a negative signal to pay.

The default parameters (`C = 6`, `L0 = 100`, `L1 = 20`, prior 12%, `q = 0.8`, 20 units per hour) are **hypothetical teaching values**. They are not estimates for the November 2026 Portuguese alert, Storm Kristin or the 2024 Valencia floods, and no figure claims a number of lives saved.

## Figures

| File | What it shows | Status |
| --- | --- | --- |
| `figures/01_chronology.png` | Depression Charles and the Valencia DANA on one axis, in hours from the start of the first red-warning period | Documented milestones; not a causal comparison |
| `figures/02_cost_loss.png` | Expected loss of protecting now and of doing nothing; threshold 7.5% | Hypothetical parameters |
| `figures/03_value_of_waiting.png` | Gain from waiting for an 80%-accurate forecast as protection degrades | Hypothetical parameters |
| `figures/04_accuracy_and_delay.png` | Longest worthwhile delay against forecast accuracy, priors 12% and 25% | Hypothetical parameters |
| `figures/05_reversibility.png` | Protect now and review, against the share of the cost still avoidable | Hypothetical parameters |

## Sources for Figure 1

- [Portuguese Government: alert declared for mainland Portugal, 10 November 2026](https://portugal.gov.pt/pt/gc25/comunicacao/noticias/governo-declara-situacao-de-alerta-em-todo-o-territorio-continental-1)
- [Portuguese Government / Ministry of Internal Administration: announcement and IPMA red warnings, 10 November 2026, 10:33](https://portugal.gov.pt/pt/gc25/comunicacao/noticias/mau-tempo-governo-declara-situacao-alerta-)
- [Portuguese Government / ANEPC: update, 11 November 2026, 13:31](https://portugal.gov.pt/gc25/comunicacao/noticias/mau-tempo-protecao-civil-apela-a-adocao-de-medidas-de-seguranca)
- [Portuguese Government: prime minister's assessment, 11 November 2026, 21:10](https://portugal.gov.pt/gc25/comunicacao/noticias/mau-tempo-prevencao-permitiu-superar-momentos-de-maior-exigencia-afirmou-o-primeiro-ministro)
- [Maldita.es: chronology of AEMET warnings and the ES-Alert message, 29 October 2024](https://maldita.es/clima/20241030/cronologia-dana-avisos-amet-ayudas-gobierno-valencia/)
- [Maldita.es: AEMET warnings and the civil-protection alert](https://maldita.es/clima/20241030/avisos-meteorologos-proteccion-civil-valencia/)

Portugal uses WET (UTC) in November. Valencia used CET (UTC+1) on 29 October 2024: summer time had ended on 27 October.
