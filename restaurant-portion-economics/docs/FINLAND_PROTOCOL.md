# Finland data and menu-matched descriptive protocol

Source: [Zenodo 20825233](https://zenodo.org/records/20825233), Finnish
school food waste (2024–25). The source reports 155 days (135 baseline,
20 intervention) and a rotating 25-day menu. Institutional catering is
not commercial restaurant demand.

The parser recognises only these documented source labels:
- PW (plate waste, kg/day and g/diner/day)
- KSW (kitchen/serving waste, kg/day and g/diner/day)
- DINERS (count)
- MENU (identifier 1–25)

A workbook may contain introductory header rows. The import scans up to
30 rows per sheet, requiring one unambiguous table. If the actual publisher
headers differ or several workbooks are present, the script fails rather
than substituting guessed data. A manual codebook audit is required to add
new aliases. A row order of 135+20 is used as provided by the record, not
derived from unverified local dates.

For each period and type, report days observed, diners, total kg, pooled
g/diner, and unweighted mean daily g/diner. Audit whether
1000 * mass(kg) / diners agrees with published per-diner values; report
discrepancies rather than silently correcting them.

Menu matching: for each of the 20 intervention rows, use the mean
baseline g/diner/day for that row's menu ID. Compare the average matched
baseline mean against the average intervention values, over days where
both exist. This standardises the *observed menu mix* but not diners,
calendar time, service changes, reporting practices, or unobserved
interventions. It is a descriptive difference, **not a causal effect**.
No p-values are reported because 155 serially dependent observations
and possible changing data definitions make naive uncertainty misleading.

The R implementation has offline synthetic tests. The publisher workbook
has not been executed in the authoring environment. Do not report real
numerical results until it passes a complete source validation.
