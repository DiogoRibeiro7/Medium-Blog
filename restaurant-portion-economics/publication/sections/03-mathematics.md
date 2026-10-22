## The contribution-margin equation

Suppose a restaurant sells N meals a month at price p. Let c be the ingredient cost of a meal and v the other variable costs attributable to serving it. Contribution per meal is m = p − c − v. Fixed costs matter for whether the business survives, but when they are genuinely the same under both alternatives they cancel out of the comparison, so I work with contribution rather than pretending to estimate net profit from accounts I do not have.

Let r be the fraction by which the default portion is cut. Not every ingredient shrinks with it. Sauces, garnishes, minimum order quantities, fixed recipe components, packaging and anything the restaurant should not reduce all stay where they are, so let s be the share of ingredient cost that the cut actually touches. Even within that share, preparing less does not always mean buying less: suppliers sell in cases, waste moves between stages, and purchasing habits adjust slowly. Let a be the fraction of the theoretical reduction that shows up in the purchasing bill.

The avoidable ingredient cost per retained customer is then

S = c × s × r × a.

The equation is deliberately plain. It claims neither that every restaurant can save this amount nor that discarded-food data supply any of its parameters. Each factor has a physical or operational meaning that a particular restaurant could measure.

Now suppose the new portion reduces the number of meals sold by a proportion l, and let I be the monthly cost of implementing the change: retraining, new menus, changes to kitchen routines. Monthly contribution before the change is N × m. After it, it is N × (1 − l) × (m + S) − I. The difference is

Δ = N × [(1 − l) × (m + S) − m] − I,

which rearranges to

Δ = N × [S − l × (m + S)] − I.

The first term is the purchasing saving the restaurant would make if demand did not move. The second removes the contribution lost on the meals that are no longer sold, including the savings that would have been made on them. The third is the cost of making the change. At this level the model is an accounting identity conditional on its inputs, not a fitted relationship between behaviour and outcomes.

Setting Δ = 0 and solving for the largest loss of customers the change can absorb gives

l* = (S − I/N) / (m + S),

provided m + S is positive. If l* is negative, the change fails even if every customer stays. If l* is positive but small, the business case rests on customer acceptance to a degree that can easily outweigh the food saving. For an operator, that threshold is more useful than any recommendation to cut portions by a fixed percentage.

To make this concrete, take an illustrative restaurant that sells 2,500 meals a month at €15 each, with €5 of ingredients and €2 of other variable cost per meal, so that each meal contributes €8 before fixed costs. Cut the default portion by 15%, let the cut touch 40% of ingredient cost, let 85% of the theoretical saving reach the purchasing bill, and allow €150 a month to implement it.

The saving per retained meal is €5 × 0.40 × 0.15 × 0.85 = €0.255. If no customer leaves, monthly contribution rises by 2,500 × €0.255 − €150 = €487.50. With 1% fewer meals the gain is about €281.13; with 2% fewer, about €74.75. With 5% fewer, the change costs about €544.38 a month. These are arithmetic scenarios, not observed outcomes.

![Change in monthly contribution against the share of meals lost, with the break-even point](figures/fig06_contribution_vs_loss.png)
*Figure 6. Change in monthly contribution for the illustrative restaurant as a function of the share of meals lost after a 15% portion cut. Hypothetical parameters.*

The break-even loss in this example is about 2.36% of monthly meals. A modest drop in demand is enough to absorb the entire saving. That does not mean a 15% cut will lose 2.36% of customers; estimating that relationship requires evidence I do not have, and the rest of this piece is largely about how much can be said without it.

The structure of l* also explains why the threshold is so tight. Ignore implementation cost for a moment and l* is roughly S/m, the saving per meal as a fraction of the contribution per meal. Ingredients are a minority of the price, only part of them shrinks with the portion, and the margin on each meal is several times larger than the saving. Losing one meal costs the restaurant its whole margin; keeping it gains a few cents.

The inputs are not Portuguese restaurant benchmarks. I chose them to make the model easy to inspect. Later I set them against US industry ratios of food cost to sales, but a business-level ratio of food and non-alcoholic beverage spending to sales is not the ingredient cost of a particular €15 dish, and treating it as one would confuse both the unit of observation and the composition of cost.

This is where the model becomes useful to someone running a restaurant. An owner does not need to accept any argument about sustainability to use it. The question reduces to two operational quantities: how much ingredient cost is actually avoidable per retained meal, and how many transactions can be lost before the change stops paying. Those two numbers, together with implementation cost, settle most of the answer.
