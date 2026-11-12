# Extreme weather warnings: decisions under uncertainty

Public reproducibility companion for Diogo Ribeiro's article, *When a False Alarm Is the Right Decision: The Mathematics and Ethics of Extreme Weather Warnings* (12 November 2026).

The editorial manuscript is maintained separately in the private [`research-articles-medium`](https://github.com/DiogoRibeiro7/research-articles-medium) repository. This public folder contains **no full manuscript**.

## Figures

| Figure | File | What it shows | Evidential status |
| --- | --- | --- | --- |
| 1 | [`01_event_timeline.png`](figures/01_event_timeline.png) | Selected milestones for Depression Charles (Portugal, November 2026), Storm Kristin (Portugal, January 2026), and the Valencia DANA (Spain, October 2024) | Source-grounded chronology, not a causal estimate |
| 2 | [`02_expected_loss.png`](figures/02_expected_loss.png) | An illustrative two-action cost–loss model, with a 7.5% threshold | Hypothetical parameters |
| 3 | [`03_vulnerability_thresholds.png`](figures/03_vulnerability_thresholds.png) | How avoidable harm changes the intervention threshold | Hypothetical parameters |
| 4 | [`04_value_of_waiting.png`](figures/04_value_of_waiting.png) | How forecast updates lose decision value as protection is delayed | Hypothetical signal and delays |
| 5 | [`05_warning_chain.png`](figures/05_warning_chain.png) | The operational steps between a forecast and protective action | Conceptual diagram |

The parameters used for Figures 2–4 are **not estimates of the November 2026 Portuguese alert, the January 2026 storm, or the 2024 Spanish floods**. No figure claims a counterfactual number of lives saved or attributes a death to a specific official.

## Regenerate the figures

Requirements: Python 3.11 or newer, NumPy and Matplotlib.

```bash
python -m pip install -r requirements.txt
python figure_reproduction.py
```

The script saves five PNGs in `figures/`. It is deterministic and does not fetch or modify external datasets. Its time labels are annotations of selected documented institutional milestones, not a comprehensive event log.

## Model logic

The two-action threshold is derived by comparing the expected loss of immediate protection with inaction:

\[
p^* = \frac{C}{L_0-L_1},
\]

where `C` is the cost of protection, `L_0` is loss if a severe event occurs without protection, and `L_1` is loss with protection. In Figure 2, the deliberately hypothetical values `C=6`, `L_0=100`, and `L_1=20` give `p*=0.075`.

The value-of-waiting illustration assumes a severe-event prior of 12%, and a later binary signal with 80% sensitivity and 80% specificity. The residual loss after delayed protection is increased with the delay. The figure compares the resulting expected loss under that signal policy against protecting immediately, thereby separating **information gain** from **lost lead time**.

## Source records for Figure 1

- [Portugal Government: nationwide alert, 10 November 2026](https://portugal.gov.pt/pt/gc25/comunicacao/noticias/governo-declara-situacao-de-alerta-em-todo-o-territorio-continental-1)
- [Portugal Government / ANEPC: 11 November midday situation update](https://portugal.gov.pt/gc25/comunicacao/noticias/mau-tempo-protecao-civil-apela-a-adocao-de-medidas-de-seguranca)
- [Portugal Government: 11 November evening assessment](https://portugal.gov.pt/gc25/comunicacao/noticias/mau-tempo-prevencao-permitiu-superar-momentos-de-maior-exigencia-afirmou-o-primeiro-ministro)
- [IPMA: extreme winds during Storm Kristin, January 2026](https://www.ipma.pt/pt/media/noticias/news.detail.jsp?f=Tempestade_kristin_portugal_continental.html&y=2026)
- [AEMET: technical study of the October–November 2024 DANA](https://www.aemet.es/en/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/detalles/episodio_dana_oct_nov24)
- [RTVE: chronology of the 29 October 2024 warning and mobile alert](https://www.rtve.es/noticias/20241031/cronologia-dana-12-horas-alerta-movil/16311521.shtml)

The manuscript contains the complete twenty-one-item source list, the mathematical derivations, and the discussion of what the available evidence cannot establish.
