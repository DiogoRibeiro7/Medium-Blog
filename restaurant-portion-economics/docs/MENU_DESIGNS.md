# Menu-design counterfactuals and customer response

A smaller default portion can be presented at an unchanged price, a discount,
or with an optional free refill. These are distinct commercial strategies:
equating them would hide price-related contributions, service costs, and
customer responses.

The R module `R/menu_designs.R` makes these accounting differences explicit.
An input portion reduction is not itself an observed reduction in food waste.
Refill uptake, ingredient savings, and lost customer volume are independent
scenario inputs, not fitted elasticities.

## Definitions

Let baseline menu price p, ingredient cost c, additional variable cost v,
monthly meals N, affected share s, portion reduction r, purchase savings
realization a, refill uptake u, restoration fraction f, refill handling
cost h, discount fraction d, demand loss l and monthly implementation I.

The retained-meal ingredient saving, after allowing for one optional
refill, is S = c*s*r*a*(1-u*f). The modified per-meal contribution is

    m_new = p*(1-d)-c-v+S-u*h,

and the monthly change is

    Delta = N*(1-l)*m_new - N*(p-c-v) - I.

The break-even customer loss is

    l* = 1 - (p-c-v+I/N)/m_new

when m_new > 0. A negative threshold means that retaining everyone would
still be insufficient. The largest tolerable discount can also be computed
without estimating a demand response.

The relative price per unit of default portion mass is
(1-d)/(1-r)-1, with the strong assumption that all default portion mass
decreases proportionally by r. It is an arithmetic indicator, not a
measured consumer value judgement.

## Evidence boundaries

The pre-existing source register includes field studies using smaller
portions, but none identifies how reducing commercial restaurant portion
size while retaining the same price affects repeat visits. Consequently,
the proposed 0%, 1%, 2%, and 5% loss scenarios are hypothetical. No paper
is translated into a customer-retention parameter.

A hypothetical refill must also be interpreted carefully. A customer
who requests a refill may restore some of the initial ingredient saving,
and handling the request incurs labor/service effort. Without actual
refill uptake, the model cannot determine which design is best.

In particular, do not equate calorie reductions with financial savings
or with nutritional adequacy. None of the scenarios automatically
satisfies individual nutritional requirements.

## Execute

From `restaurant-portion-economics/`:

```sh
Rscript tests/test-menu-designs.R
Rscript scripts/analyse-menu-designs.R
```

Two CSV files are written under ignored `outputs/`. They are synthetic
accounting comparisons, not empirical restaurant KPIs or causal estimates.
