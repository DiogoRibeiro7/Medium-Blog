# Census 2021 directed residential mobility: access and intake specification

**Status (16 November 2026): published municipality-level migration categories
verified; named residential origin–destination data not requested or received.**
This is an actionable research-data specification, not evidence of
Porto-to-suburb relocation. The repository must not host or fabricate
individual Census records.

## 1. Why the official data could answer the question

[INE's Census 2021 individual questionnaire, question 10](https://censos.ine.pt/scripts/censos_css_js/quest/EN_Q_IND_Censos2021_EN_v03.pdf)
asks for the person's residence on **31 December 2019**. When the response is
"another municipality", the questionnaire explicitly requests the *name of
that municipality* and parish. The Census reference date is **19 April 2021**.
The collected variable is therefore more specific than the categories in
[INE's publicly indexed population-by-migration-category indicator 0011631](https://www.ine.pt/xurl/indx/0011631/PT)
or the original [Porto municipal archive](https://dadosabertos.cm-porto.pt/dataset/indicadores-relativos-populacao-residente-no-porto-resultados-definitivos-censos-2021).

The existing archive provides **15,724** current Porto residents who lived in
another Portuguese municipality on the earlier date and **124,618** for the
whole Porto Metropolitan Area (AMP). Neither number identifies origin.
The required cross-tabulation has not been located in the inspected public
tables. **Collection of named previous municipality does not guarantee access
to it**, or that INE will approve a geographically detailed table.

### Verified public-category data, not residential origin–destination data

The official [INE Census 2021 indicator 0011631](https://www.ine.pt/xurl/indx/0011631/PT)
has now been independently queried for **each of the 17 AMP destination
municipalities**. It exposes the resident population by sex and by eight
residence/migration categories, including category `33` (*previously resident
in another Portuguese municipality*). It provides **no previous-municipality
name or code**, and hence no Porto-to-Gaia, Porto-to-Maia or other directed
residential origin–destination cell.

Source evidence was fetched read-only in [GitHub Actions run
43897407323](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43897407323),
attempt 3, which validated **17/17 municipalities**. Exact raw 2021 JSON
responses, the original SHA-256-bearing receipt and a losslessly compressed
copy of the matching official metadata are pinned under
`data/ine_0011631_amp17_2021_source/`. No synthetic observations enter the
published data.

The offline audit in `ine_residential_municipal_audit.py` checks the official
indicator/period/geography/sex/category identities, all 24 cells per
municipality, subgroup accounting, source checksums and **two independently
published margins** from the Porto municipal archive:

| Current residents in 2021 who lived in another Portuguese municipality in 2019 | People |
| --- | ---: |
| Porto (destination `1312`) | 15,724 |
| Entire AMP (17 destinations) | 124,618 |
| Other 16 destinations, combined | 108,894 |

The verified derived panel is
`data/amp_census2021_municipal_residential_categories.csv`; its companion
`_provenance.json` records the hashes of every official source and the frozen
Porto archive. The `moved_from_other_municipality_pct_eligible` denominator
is the municipality's 2021 usual residents who were already born at the
31 December 2019 reference date. It is a **retrospective in-mover category
share**, not an annual migration rate, a count of departures, an origin–
destination flow or a housing-price response.

To audit the committed sources without network access, from this study folder:

```bash
python ine_residential_municipal_audit.py \
  --capture-directory data/ine_0011631_amp17_2021_source \
  --output-csv /tmp/ine-0011631-amp17-categories.csv \
  --output-provenance /tmp/ine-0011631-amp17-provenance.json
```

The 17 destination counts reinforce the validity of the existing marginal
controls but **cannot narrow the sharp identification bounds on Porto's
directed residential exchange**: every destination's category `33` includes
origins anywhere else in Portugal. The custom INE cross-tabulation specified
below remains necessary for that question.

## 2. Two documented routes for obtaining an authorized table

### A. Request an additional statistical tabulation from INE

INE has explicitly described a [customer statistical-information request
service](https://www.ine.pt/ine_novidades/semin/FI_RIIBES_47/files/assets/basic-html/page10.html)
for more detailed Census data and additional cross-tabulations. Its published
[request form link](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_ped_informacao)
is the first route. **The live form was not accessible during this research
session**; this link is documented by INE, not verified as operational now.
Do not submit until it is accessible or INE confirms a current alternative.

Ask whether the table can be provided without cost or under an academic
research arrangement **before** commissioning any paid production. Never
assume that an information request is free.

### B. Ask about the accredited scientific-research route

INE has described an [INE–FCT–DGEEC research-access arrangement](https://www.ine.pt/ine_novidades/semin/FI_RIIBES_51/files/assets/basic-html/page6.html)
for accredited university or research-institution researchers, including
approved anonymized tables and secure microdata access. The [April 2026 INE
newsletter](https://www.ine.pt/ine_novidades/semin/INEWS65/index.html)
confirms ongoing safe-centre access, including activities in Porto and
improvements to its research infrastructure. These documents **do not
establish that Census 2021 residence histories are available in a particular
microdata product**, or that every university-affiliated applicant is
automatically eligible. Institution, project and output approval must be
confirmed with the current programme coordinator.

Prefer a small privacy-safe *aggregate table* to person-level data.
The public manuscript requires only aggregate observations.

## 3. Precisely requested data: a 17-destination × 18-origin table

**Universe:** persons habitually resident in the 17 AMP municipalities on
19 April 2021 who had been born by 31 December 2019 and lived in *another
Portuguese municipality* on that earlier date, according to Census 2021 Q10.

**Destinations:** all 17 legally defined AMP municipalities using their
official 2021 four-digit DICO codes, including Porto `1312`.

**Origins:** those same 17 named municipalities, plus one grouped category
`REST_PT` for **every other Portuguese municipality outside AMP**. This is a
research-delivery grouping, not an INE geography code. It is necessary for
reconciling destinations without making the origin matrix national in size.

**Measure:** number of resident *individuals* at each domestic municipality
origin → AMP destination pair. Same-municipality pairs are excluded from
"another municipality" and represented as structural-zero diagonals in the
normalized delivery. **Moves from abroad are not included.** Temporary
commuting, place of birth, and previous residence at a different reference
date must not enter the requested measure.

For all 17 destinations combined, the 2021 intermunicipal category should
reconcile to **124,618**. For Porto as a destination it should reconcile to
**15,724**. The other 16 destinations jointly should reconcile to
**108,894**. If INE's custom release uses a different population universe,
vintage or treatment of non-response, that discrepancy must be documented,
not silently forced into the existing totals.

The source should include:

1. Origin and destination municipality identifiers, geographic version and
   any 2019↔2021 boundary reconciliation.
2. Defined reference dates, persons versus households, eligibility and
   treatment of non-response and multiple moves.
3. Cell disclosure status, suppression rules, any rounding/perturbation
   method, and whether marginal totals are independently protected.
4. Citation, source release and extraction dates, access licence and whether
   derived municipal aggregates may be shared in a public repository.
5. The official source's own aggregate control totals and any quality flags.

If the 18-origin grouping is not releasable, ask INE for the **smallest
permitted safe alternative**, such as Porto versus rest of AMP versus rest
of Portugal, broken down by the 17 destinations. Do not request personal
identifiers, geocoded addresses or small-cell disclosure.

## 4. Request text, for review and submission (not sent)

**Assunto: Pedido de apuramento estatístico especial — Censos 2021,
mobilidade residencial entre municípios da Área Metropolitana do Porto**

> Exmos. Senhores,
>
> Encontro-me a desenvolver um estudo científico sobre a evolução dos custos
> da habitação e a mobilidade residencial na Área Metropolitana do Porto.
> Pretendo averiguar a disponibilidade de um apuramento agregado e
> anonimizado dos Censos 2021, com base na pergunta 10 do questionário
> individual: município de residência a 31 de dezembro de 2019, cruzado com
> município de residência à data dos Censos (19 de abril de 2021).
>
> Solicito, se for possível, o número de pessoas residentes em 2021 em cada
> um dos 17 municípios da Área Metropolitana do Porto que, em 31 de dezembro
> de 2019, residiam noutro município português. Pretendia distinguir os
> 17 municípios da AMP como origens e agregar os restantes municípios do país
> numa única categoria. Não necessito de microdados individuais nem de
> quaisquer elementos de identificação pessoal.
>
> A finalidade inicial é verificar os movimentos residenciais efetivos entre
> o Porto e os municípios envolventes, sem pressupor que tenham sido causados
> pelo preço da habitação. Seria importante conhecer os códigos territoriais,
> as regras de confidencialidade e os eventuais valores suprimidos, de modo
> a permitir a reconciliação com os totais publicados (15 724 residentes
> no Porto e 124 618 na AMP na categoria de proveniência de outro município).
>
> Agradecia a indicação de disponibilidade, condições de reutilização,
> eventuais custos e, se existir, a possibilidade de acesso gratuito para
> investigação científica. Se o apuramento solicitado não puder ser divulgado,
> agradecia informação sobre a desagregação territorial mais próxima que
> possa ser disponibilizada e os procedimentos atuais para apresentação
> de um pedido ao abrigo de um protocolo de investigação.
>
> Com os melhores cumprimentos,
>
> Diogo Ribeiro

Before submission, confirm the institutional affiliation and preferred
contact details. Do not assert institutional authorization without it.

## 5. Normalized research-intake contract

Any approved source must be documented and converted **locally**, preserving
its original bytes and an immutable checksum. The normalizer is not included
yet because the delivery format is unknown. The validator accepts a canonical
UTF-8 CSV with *exactly* these columns:

| Field | Type | Rule |
| --- | --- | --- |
| `origin_dico` | string | One of 17 AMP four-digit DICO codes, or `REST_PT` |
| `destination_dico` | string | One of 17 AMP four-digit DICO codes |
| `count` | string representation of integer or empty | `0` or canonical positive integer if observed; empty only if suppressed |
| `disclosure_status` | enum | `observed`, `suppressed`, or `structural_zero` |

The complete grid has **18 origins × 17 destinations = 306 cells**. Its
17 same-municipality diagonal cells have `structural_zero` and count `0`;
no off-diagonal cell may be marked structural zero. An actually observed
zero is `observed,0`, **not** a suppression.

The companion JSON must contain this structure; replace the source fields
with the approved release's metadata, and compute the SHA-256 of the exact
normalized CSV:

```json
{
  "contract_version": "census2021_residential_od_amp_v1",
  "publisher": "INE",
  "reference_start": "2019-12-31",
  "census_reference": "2021-04-19",
  "population_universe": "2021 usual residents with previous residence in another Portuguese municipality",
  "origin_scope": "AMP17_PLUS_REST_PT",
  "destination_scope": "AMP17",
  "unit": "persons",
  "domestic_previous_residence_only": true,
  "source_title": "<INE release title>",
  "source_citation": "<INE source URL, reference or approved access identifier>",
  "license": "<approved permitted reuse>",
  "disclosure_policy": "<official cell suppression/rounding policy>",
  "source_csv_sha256": "<64-character SHA-256 of normalized CSV>"
}
```

Run, once a real authorized release exists:

```bash
python residential_od_intake.py \
  --source-csv /secure/location/approved_aggregate_od.csv \
  --metadata-json /secure/location/approved_release_metadata.json \
  --output-json /tmp/porto-residential-od-audit.json
```

The script validates complete geographic coverage, source metadata/hash,
cell statuses, integer counts, and the two official destination margins.
When some cells are suppressed, it calculates **logical intervals**, assigning
the unobserved residual only to suppressed cells, including cases where
the destination margin identifies their aggregate sum exactly. It never
replaces suppressed counts with zero or invents individual origin pairs.
Its diagnostic is a **data-validation artifact, not a causal estimate**.

**Tests use synthetic tables only in temporary directories.** No simulated
origin–destination table, real microdata or hypothetical relocation value
is committed in the data directory as if it were observed.

## 6. Interpretation and stopping rules

If INE cannot provide sufficiently detailed counts, preserve the existing
sharp bounds and report that the direction of residential exchange remains
unidentified. If actual OD counts are released, report observed municipality
flows, suppression uncertainty and sensitivity to geography and reference
dates. A retrospective cross-tab tells us where 2021 residents were living
on an earlier date. It does not count intervening moves or prove *why* anyone
moved. A displacement effect needs separate economic outcomes and an
identification strategy.
