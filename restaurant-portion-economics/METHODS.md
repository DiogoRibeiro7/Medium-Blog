# Analysis protocol, version 0.1

## Target claim

Determine ranges of procurement savings and customer-volume changes under
which reducing default portion size (without changing menu price) could
improve contribution margin. We do **not** claim to estimate an optimal
portion size from aggregate food-waste records.

## Distinct quantities

- Purchased inputs, prepared food, plated quantity, food consumed, and
  leftovers are distinct and should not be conflated.
- Kitchen/preparation waste differs from plate waste and spoilage.
- A 15% smaller plate is not necessarily a 15% lower ingredient bill.
- Observational school-canteen data are not representative of Portuguese
  commercial restaurants.
- Demand elasticity is missing. Price remaining unchanged does not imply
  unaffected demand.
- Reduced caloric content is not intrinsically beneficial for every customer;
  meal composition and nutritional adequacy matter.

## Stage 1: external source audit

Check provenance, date, licensing and file hashes. Preserve raw input files
outside Git; export field names, R types and missing counts first. Do not
arbitrarily guess field units or join unrelated observational populations.

## Stage 2: deterministic economic scenarios

Compute monthly contribution differences from the explicitly hypothetical
cost structure in the README. Vary portion reduction and customer loss to
show the sign and magnitude of the margin impact. Report break-even
customer-volume loss. Treat procurement savings pass-through as independent
from portions left uneaten.

## Stage 3: empirical calibration (future)

Only after the URBAN-WASTE codebook and files are inspected should specific
waste fields be summarized by establishment and waste stage. External
reference statistics may constrain plausible waste reductions but cannot
identify causality of reduced portions. Do not use post-treatment selection
or conflate institution-level and individual-level records.

## Review criteria

1. Reproducible raw sources and hash checks.
2. Schema and units audited; missingness reported.
3. No synthetic numbers represented as observations.
4. Profitable scenarios identified with sensitivity boundaries.
5. No empirical claims of actual restaurant margins absent valid sales data.
