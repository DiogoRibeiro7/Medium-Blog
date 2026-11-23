# Finland 1986 and Germany 2008: tax-policy component evidence

This audit follows the [merged 20-event feasibility study](IDENTIFICATION_FEASIBILITY.md).
These are the **only two** previously eligible Hope–Limberg 2-SD tax-cut
onsets with strict geometric support in three pre-reform WDI macro means
through year five. The earlier legislative review covered neither case.
Geometric donor overlap does **not** imply causal positivity.

The author treatment is a multi-tax composite annual indicator. **Neither
case below is independently proven to be the legal cause of its author
flag**, and no new causal effects are estimated.

## Finland, author flag 1986

The enacted annual tax-statute identifier is
[Finlex 997/1985](https://www.finlex.fi/fi/lainsaadanto/1985/997).
The 1985 government [bill HE 114/1985](https://www.edilex.fi/he/fi19850114.pdf)
contains the proposed sections (it is a *proposal*, not proof of the
final statutory wording). Implementation is separately corroborated
by Finnish Ministry of Finance official Lasse Aarnio, writing in the
[Bank of Finland Bulletin, April 1986, pp. 26–32](https://publications.bof.fi/items/005ef29d-d4bc-405a-879d-b3611dde7332).

| Instrument | Before | 1986 or later | Qualification |
|---|---:|---:|---|
| Central-government corporate income-tax rate | 43% | **33%** in 1986 | 10-percentage-point rate reduction |
| Personal wealth-tax exemption threshold | FIM 275,000 | **FIM 800,000** | Threshold increase with unchanged maximum 1.7% marginal wealth-tax rate |
| Corporate advance collection | 43% | **43%** in 1986 | Final statutory rate was 33%, but advances stayed at 43%; excess refunded in 1987 |
| Capital-gains/sales-proceeds base | Narrower | **Broadened** from 1986 | Countervailing tax-base *tightening*, not universal tax relief |

The capital-gains law
[Finlex 1038/1985](https://www.finlex.fi/fi/lainsaadanto/1968/360),
dated **20 December 1985**, entered into force **1 January 1986**,
with old rules preserved for earlier dispositions.
The [Bank of Finland, January 1987](https://publications.bof.fi/bitstream/handle/10024/48767/Bofbul_1987-01.pdf?isAllowed=y&sequence=1)
confirms delayed excess advance-payment refunds.
The **exact promulgation date** for annual statute 997/1985 was not
independently verified in this review; the bill's 18 September 1985
signature date is *not* used as its enactment date.

## Germany, author flag 2008

The [Bundestag legislative archive XVI/0221](https://dserver.bundestag.de/gm/16/160221.pdf)
documents the **Unternehmensteuerreformgesetz 2008**, enacted on
**14 August 2007**, published in BGBl. I 2007 p. 1912. The
[Federal Ministry of Finance monthly report, May 2008, pp. 45–47](https://www.bundesfinanzministerium.de/Monatsberichte/2001-2016/Inhalte/Monatsbericht-Archiv-Downloads/2008/Monatsbericht_Mai_2008.pdf?__blob=publicationFile&v=1)
confirms the main reform entered into force **1 January 2008**.

| Instrument | Before / reference | Documented scope | Caveat |
|---|---|---|---|
| Corporate income-tax rate, KStG §23 | 25% | **15%**, 2008 onward | Statutory corporation rate, not the top-1% effective tax rate |
| Trade-tax assessment factor | Revised rate | **3.5%**, 2008 | Not Germany's final total effective tax rate |
| Interest-deduction limitation (Zinsschranke) | Restriction in same package | 2008 | Some base-broadening offsets relief |
| Private capital-income withholding | New flat regime of 25% | **January 2009** | A later legislative stage, not verified as part of the author's *2008* composite cut |

The Ministry forecast approximately **€6.6 billion** lower tax
receipts for the **whole package** in 2008, *not* an estimated
causal effect on pretax national-income inequality.

## Fixed WID/WDI/DPI results

| Historical event gate | Finland 1986 | Germany 2008 |
|---|---:|---:|
| Existing h=0..5 three-WDI donor-hull support | Yes | Yes |
| Original surviving donors at h=5 | **8** | **16** |
| Original donors with complete pre-five-year DPI | **7** | **15** |
| Additional author-coded major cut through h=5 | No | No |
| Pre-reform top-1% income-share slope gap vs selected donors | **−0.135 pp/year** | **+0.496 pp/year** |
| Verified statute-to-composite-flag assignment | **No** | **No** |
| New identified policy-specific causal ATT | **No** | **No** |

The German pretrend slope difference is especially material and
cannot be hidden behind an observed three-WDI hull certificate.
The 2008 reform also coincides with the global financial crisis,
complicating the counterfactual. Finland's tax-base expansion and
cash-collection lag likewise complicate a single-policy interpretation.

## Reproducible implementation

The [eight-component original-source register](policy_component_sources.csv)
contains official links, instrument-level before/after values where
comparable, exact statutory-period scope, collection cautions, and
an explicit **unverified** author-flag attribution for *every* row.

The typed [audit module](policy_component_candidates.py) joins that
register to the **unchanged** 20-event identification-feasibility
artifact from merged PR #766. It rejects changed donor sets, new
causal-status claims, unknown source domains, nonfinite rates, reused
component identifiers, invalid dates and unverified tax years.
The produced files are:

- policy_component_evidence.csv — all eight documentary component rows
- supported_reform_candidate_gates.csv — the two original event gates
- policy_component_audit.json — counts, limitations and source SHA-256

Run from the repository's neoliberalism-evidence-synthesis directory:

~~~bash
python -m causal_tax_cuts.policy_component_candidates \
  --components causal_tax_cuts/policy_component_sources.csv \
  --prior-events data/tax-study/identification-feasibility/policy_identification_events.csv \
  --prior-audit data/tax-study/identification-feasibility/policy_identification_audit.json \
  --output-dir data/tax-study/policy-component-candidates
~~~

These **two newly reviewed policy contexts** raise the combined
statutory-or-contemporaneous documentary coverage from five to
**seven of 20** eligible onsets, with distinct source classifications:
Finnish annual law identifiers and contemporary ministry confirmation,
versus Germany's enacted law docket and federal ministry report.
Neither establishes a unique intervention or causal treatment date.

Next, to estimate an actual policy-specific effect, separately
predefine taxpayer exposure, enactment and anticipation timing,
cointerventions, valid controls and an estimand using more
appropriate tax/administrative data. Do **not** present the original
binary composite as uniquely attributable to the listed statutes.
