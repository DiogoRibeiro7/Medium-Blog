# Which Rules of the Monty Hall Game Change the Decision? A Value-of-Information Analysis of the Host's Protocol

## Part 2 of 4. The host's tie-break, his knowledge, the placement of the car and his offer policy, taken one at a time: which ones move the probability, which ones move the decision, and how a contestant could learn the difference from past episodes

The first article in this series followed the Monty Hall problem from Bertrand's boxes in 1889 to the death of Steve Selvin in September 2026, and ended by separating three layers that the public argument of 1990 had run together: the number two-thirds, the model that produces it, and the evidence people used to convince themselves. This article is about the middle layer. The model is a description of the host, of what he knows and what he does, and every disagreement about the problem since 1975 has turned out to be a disagreement about some part of it. Monty Hall raised the first objection himself, in a letter to Selvin that pointed out that his show did not let contestants trade once a box had been opened. Martin Gardner and Persi Diaconis told *The New York Times* in 1991 that the question cannot be answered without the host's motivation. Morgan, Chaganty, Dahiya and Doviak turned the host's preference between two goat doors into a parameter. Jeffrey Rosenthal named a host who does not know where the car is. Richard Gill argued that the problem is really a game between the host and the contestant.

Most treatments of these variants do what the literature did: they take a host, compute the probability that switching wins, and move on to the next host. The result is a catalogue, and catalogues make every variant look equally important. The question a decision analyst would ask is different. The contestant does not need a probability; he needs to know whether to switch. Some features of the host change the probability without ever changing the answer to that question, and some change the answer. A data scientist meets the same distinction every time a model is criticised: a mis-specified detail that shifts a predicted probability but never changes the action taken on it is a problem for the report, not for the decision, and the effort spent checking the model should go where the decision is sensitive.

This article applies that test to the host. It writes the host's rules as a model with four components (how he breaks a tie between two goat doors, whether he knows where the car is, where the car is likely to be, and when he offers the switch at all), derives the probability that switching wins under each, and then asks, for each component separately, whether it can make staying the better choice, whether a contestant has a strategy that protects him without knowing it, and how much it would be worth to learn it. The answer is that only one of the four can turn the advice around in a way no strategy protects against, and it is the one Monty Hall demonstrated at his dining-room table in 1991: an offer that depends on whether the contestant's first pick was right. The last part of the article treats that component as something to be learned from data, and asks how many past episodes a contestant would need to watch before he could tell which kind of host he faces. Every number is printed by a script in the series repository, linked at the end, which also checks each closed-form expression against exact enumeration with rational arithmetic in 366 parameter settings.

## A host with four dials

Fix the contestant's first pick as door 1; nothing below depends on the label. The car is behind door 1, 2 or 3 with probabilities p₁, p₂ and p₃. The host is then described by four quantities.

The **tie-break** q is the probability that the host opens door 3 rather than door 2 when the car is behind door 1, so that both other doors hide goats and he is free to choose. The textbook problem sets q = 1/2.

The **knowledge** φ is the probability that the host knows where the car is. With probability 1 − φ he opens door 2 or door 3 at random, and with probability one half that door hides the car. A game in which the car is revealed has no decision left in it, so it is treated as void; everything that follows is conditional on a goat being shown, which is also how Rosenthal's "Monty Fall" problem is posed (Rosenthal, 2008). The textbook problem sets φ = 1.

The **placement** (p₁, p₂, p₃) is the distribution of the car. The textbook problem makes it uniform.

The **offer policy** (a, b) gives the probability that the host, after opening a goat door, offers the switch at all: a when the contestant's door hides the car and b when it does not. A host who never offers leaves the contestant with his first pick. The textbook problem sets a = b = 1, so the offer is compulsory.

The contestant observes which door was opened, that it hid a goat, and whether he was offered the switch. Everything he can know about where the car is comes from the probability of that observation under each hypothesis about the car. Bayes' theorem in odds form says that, after the host opens door 3 and offers the switch, the posterior odds that the car is behind door 2 rather than door 1 are the prior odds times the ratio of the probabilities that the host would have done exactly that in each case:

```
odds(door 2 : door 1 | host opens 3, offers) = (p₂ / p₁) × P(opens 3, offers | car at 2) / P(opens 3, offers | car at 1)
```

Each of the four dials enters that expression in a different place. The tie-break and the knowledge change the probability that door 3 is the door opened. The placement changes the prior odds. The offer policy changes the probability that the switch is offered at all. That is why they behave so differently, and it is also why they can be studied one at a time: set three to their textbook values and move the fourth.

Written out with all four dials at once, and with o for the probability that an ignorant host offers the switch after a goat appears, the probability that switching wins after door 3 is opened and the switch is offered is

```
                                         p₂ [φ b + (1 − φ) o / 2]
P(switch wins | door 3, offer) = ─────────────────────────────────────────────────────
                                 p₂ [φ b + (1 − φ) o / 2]  +  p₁ [φ q a + (1 − φ) o / 2]
```

The two brackets are the probabilities of the observation "door 3 opened and the switch offered" when the car is behind door 2 and behind door 1. A knowing host contributes b in the first case, because door 3 is the only goat door he can open, and q·a in the second, because he opens door 3 only if his tie-break says so and then offers only with probability a. An ignorant host contributes the same o/2 to both, which is why his share of the games carries no information about the car. Every result below is a special case of this one expression, and the script checks it against brute-force enumeration at two hundred random combinations of the dials.

## The tie-break: a probability that never changes the decision

With a knowing host, uniform placement and a compulsory offer, the host opens door 3 with certainty when the car is behind door 2, and with probability q when it is behind door 1. The posterior probability that switching wins is therefore

```
P(switch wins | door 3 opened) = 1 / (1 + q)
P(switch wins | door 2 opened) = 1 / (2 − q)
```

which is the result Morgan and his co-authors published in *The American Statistician* in 1991 (Morgan et al., 1991). At q = 0, a host who never opens door 3 by choice, switching after door 3 wins with certainty and switching after door 2 wins with probability one half. At q = 1/4 the two probabilities are 4/5 and 4/7. At the textbook value they are both 2/3, and at q = 1 the picture of q = 0 is reflected. The left panel of Figure 1 draws both curves.

Neither curve goes below one half, and the reason is visible in the odds formula: the likelihood ratio of the host's action is 1/q after door 3 and 1/(1 − q) after door 2, and neither can be smaller than one. Staying is never strictly better than switching, whatever the tie-break, and at the extreme value it is merely as good. A contestant who switches every time wins with probability 2/3 for every value of q, because the tie-break only moves wins between the games in which door 2 opens and those in which door 3 opens. A contestant who knew q exactly and used it, staying whenever the conditional probability of switching fell to one half and switching otherwise, would also win with probability 2/3. Knowing q is worth nothing.

That last sentence can be made precise with the expected value of perfect information, the standard measure in decision analysis of what it is worth to resolve an uncertainty before acting (Raiffa and Schlaifer, 1961; Howard, 1966). It is the expected payoff of a decision-maker who learns the uncertain quantity and then acts optimally, minus the expected payoff of the best decision he can make without learning it. With a uniform prior on q, the script finds 0.6667 in both cases, an expected value of perfect information of exactly zero.

The tie-break makes a sharp illustration of the difference between information about the state of the world and information that is worth something. Measured by the mutual information between the door the host opens and the location of the car, a host with q = 0 is more informative than the textbook host: his choice of door carries 0.918 bits about where the car is, against 0.667 bits at q = 1/2 and 0.709 at q = 1/4. A contestant facing the predictable host learns more. He does not win more, because everything extra he learns concerns games in which his action was already right. The 1991 dispute between vos Savant and Morgan's group was, in this light, a dispute about a quantity that matters for a probability statement and has no consequence for the decision Whitaker's letter asked about. Both sides were right about their own question. Gill made the same point in 2011 by insisting that the problem asks for an action, not a probability (Gill, 2011); the value-of-information calculation puts a number on it, and the number is zero.

The same separation shows up if the contestant is scored the way a forecaster is. Suppose the true tie-break is q = 0 but the contestant assumes the textbook q = 1/2, and so announces, after either door opens, that switching wins with probability 2/3. A second contestant knows q and announces the correct conditional probabilities, 1 after door 3 and 1/2 after door 2. Both switch every time and both win two games in three. As forecasters, though, they are far apart: the first one's expected log loss is 0.637 nats per game against 0.462 for the second, his Brier score is 0.222 against 0.167, and his stated probability is off by 0.222 on average. At q = 1/4 the gap is smaller, a log loss of 0.637 against 0.607, and at q = 1/2 it vanishes. A model can be badly calibrated about the very event it is used to decide and still lead to the best decision, as long as its errors never move the probability across the threshold where the action changes. That is not an argument against calibration, which matters whenever the probability itself is reported, priced or combined with other information. It is an argument for checking, before investing in better calibration, whether the decision sits near the threshold at all.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/monty-hall/part2/figures/fig1_components.png)

*Figure 1. Probability that switching wins, given what the contestant has seen, as three of the host's dials move with the others at their textbook values. Shaded: the region where staying would be better.*

## The host's knowledge: the same pattern, with a cost in void games

Now let the host know where the car is only with probability φ, keep the fair tie-break and the compulsory offer, and condition on a goat being shown. If the car is behind door 1, door 3 opens with probability φ·q + (1 − φ)/2, and it hides a goat. If the car is behind door 2, door 3 opens with probability φ + (1 − φ)/2, again revealing a goat. If the car is behind door 3, a knowing host never opens it and an ignorant one opens it half the time, revealing the car; that branch is void. The probability that switching wins after a goat appears behind door 3 is therefore

```
P(switch wins | goat behind door 3) = (1 + φ) / (2 (1 + φ q))
```

which is (1 + φ)/(2 + φ) with the fair tie-break. It runs from 1/2 for the ignorant host, Rosenthal's Monty Fall, through 5/9 at φ = 1/4, 3/5 at φ = 1/2 and 7/11 at φ = 3/4, to 2/3 for the knowing host. The middle panel of Figure 1 draws it. The ignorant host also voids a third of all games by opening the car's door, a quarter at φ = 1/4, a sixth at φ = 1/2 and none at φ = 1.

The worst case over both knowledge and tie-break is one half exactly, reached when the host always opens door 3 if he can (q = 1), whatever φ is. Once again staying is never strictly better, switching is the best response for every host in the family, and with a uniform prior on φ the expected value of perfect information is zero: the script finds a conditional win rate of 0.5945 for the contestant who learns φ and for the one who simply switches. The host's knowledge is the part of the problem the letters to *Parade* got wrong, since nearly all of them computed the ignorant host's answer for a host the question said was knowing, but even for a contestant who cannot tell which host he faces, the advice to switch never costs anything. It only gains less when the host knows less.

The void games deserve one more sentence, because they are the first appearance of a theme that the third article in this series develops at length. Conditioning on a goat is not innocent. The ignorant host's games in which the car appears are removed from the data the contestant reasons about, and the removal is informative: it happens only when the car was behind the door he opened, never when it was behind the contestant's own. The answer of one half for Monty Fall is the answer for the games that survive that selection.

## The placement: the decision can turn, but a random first pick protects it

The third dial is the first that can push the probability below one half. With a knowing host, the fair tie-break and the compulsory offer, the probability that switching wins after door 3 opens is

```
P(switch wins | door 3 opened) = p₂ / (p₂ + q p₁)
```

If the car is more likely to be behind the contestant's door than behind the others, switching can be the wrong move. In the right panel of Figure 1 the two other doors are equally likely and the weight on door 1 varies; the probability that switching wins after either door opens is then one minus that weight, and it falls below one half as soon as the contestant's door holds more than half the probability. With p = (3/5, 1/5, 1/5), a contestant who has picked door 1 and always switches wins with probability 2/5, against 3/5 for staying. With p = (1/2, 3/10, 1/5) the two doors behave differently: switching after door 3 wins with probability 6/11 and after door 2 with only 4/9, so the best response is to switch after door 3 and stay after door 2, which wins 11/20 of the time against 1/2 for always switching.

Two features set placement apart from the offer policy that comes next. The first is that the contestant controls the variable it interacts with. Placement matters only in combination with the first pick, and a contestant who chooses his first door uniformly at random and then always switches wins with probability exactly 1 − E[p of the picked door] = 2/3, whatever the placement is. That is half of the minimax solution of the game Barry Nalebuff described in 1987 and Gill defended in 2011: the host hides the car uniformly and breaks ties with a fair coin, the contestant picks uniformly and always switches, and neither can push the probability away from 2/3 by deviating (Nalebuff, 1987; Gill, 2011). The random first pick is a protective strategy. It cannot be exploited, because there is nothing about the contestant's choice for the host to anticipate.

The second feature is that knowing the placement is valuable, and all of the value is upside. A contestant who knows p picks the least likely door and switches, and wins with probability one minus the smallest of the three probabilities: with p = (1/2, 3/10, 1/5), picking door 3 and switching wins 4/5 of the time. Averaged over placements drawn uniformly from the simplex, a Dirichlet(1, 1, 1) prior, the informed contestant wins 8/9 of the time against 2/3 for the uninformed one, an expected value of perfect information of 2/9, which a Monte Carlo average over 400,000 placements reproduces as 0.222. This is a real gain, and it is the kind of gain that a contestant who had studied where the show's staff tended to put the car could have pursued. It never makes the protective strategy worse than its promise.

## The offer policy: the only dial that breaks the advice

The last dial is the one Monty Hall used in front of John Tierney in July 1991, when he opened the contestant's door at once whenever the first pick was wrong and offered a switch, with money on the table, only when it was right (Tierney, 1991). Write a for the probability that the host offers the switch when the contestant's door hides the car and b for the probability when it does not. With uniform placement, the contestant's door hides the car with prior probability 1/3, so the posterior probability that switching wins, given that the switch has been offered, is

```
P(switch wins | offer) = (2/3) b / ((2/3) b + (1/3) a) = 2b / (2b + a)
```

The tie-break drops out entirely, and the script confirms that the formula holds for every tie-break tested. Figure 2 shows the probability over the whole square of offer policies. Along the diagonal a = b, the host whose offer does not depend on the contestant's luck, it is 2/3, and the textbook host sits at the top-right corner. Above the diagonal the host favours contestants who picked wrong and switching becomes more attractive, up to certainty for a host who only ever offers to people who picked a goat. Below the diagonal the host favours those who picked right, and the probability falls, reaching one half on the line b = a/2 and zero for the trap host who offers only to contestants holding the car. A sparing host who offers rarely and leans towards right picks, say a = 0.3 and b = 0.1, makes switching win only 40 per cent of the offered games, and a contestant who always switches against him wins 0.30 of all games against 1/3 for one who always stays. A slightly different sparing host, a = 0.3 and b = 0.2, gives 4/7 and makes switching right again.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/monty-hall/part2/figures/fig2_offer_policy.png)

*Figure 2. Probability that switching wins, given that the host offers the switch, over the host's offer policy. Above the solid line switching is better; below it, staying is. The marked points are the five hosts used later in the article.*

This dial differs in kind from the other three. The tie-break and the knowledge cannot push the probability below one half. The placement can, but a random first pick neutralises it. The offer policy can push the probability anywhere between zero and one, and nothing the contestant does before the offer protects him, because the host decides after seeing the contestant's choice. The offer is itself an observation, and its likelihood ratio, b/a, is controlled by the person who knows where the car is.

The value of learning the offer policy depends heavily on what the contestant believes beforehand, and the numbers show it. With independent uniform priors on a and b, the best uninformed rule is to switch, which wins half the time on average, and learning the host's policy raises that to 0.528, an expected value of perfect information of only 0.028. With a prior suited to a thrifty host, one who tends to offer when the contestant has the car (a ~ Beta(3, 1)) and rarely otherwise (b ~ Beta(1, 3)), the best uninformed rule is to stay, which wins 1/3, and perfect information is worth 0.030. With the mirror-image generous prior the best uninformed rule is to switch and information is worth almost nothing, 0.0003. These averages understate what is at stake, because they average over hosts. Against the particular trap host Hall played in his dining room, the textbook advice wins nothing at all while staying wins a third of the games; the regret of following the advice is the whole of the contestant's chance. The expected value of perfect information is small under the uniform prior because most hosts in the square are harmless or helpful. The danger is concentrated in a corner, and the corner is where a host who wants to keep his cars would choose to live.

## Knowledge is what makes the offer dangerous

The four dials interact, and one interaction is worth spelling out because it shows which dial does the damage. A host can only make his offer depend on whether the contestant's first pick was right if he knows where the car is. An ignorant host who opens a door at random and then offers the switch with some fixed probability gives the contestant no information through the offer, because he does not have any to give. Mix the two: let the host know where the car is with probability φ, play the trap policy (a = 1, b = 0) when he knows, and offer the switch every time a goat appears when he does not. After door 3 opens and the switch is offered, the probability that switching wins is

```
P(switch wins | door 3 opened, offer) = (1 − φ) / (2 − φ)
```

It is one half for the ignorant host, 3/7 at φ = 1/4, 1/3 at φ = 1/2, 1/5 at φ = 3/4 and zero for the host who always knows. The knowledge dial, harmless on its own, is the precondition for the only harmful dial. A host's ignorance dilutes his trap, and a contestant who believes the host is often guessing should be correspondingly less afraid of the offer. This is also the most charitable reading of the 1990 letters to *Parade*: their authors computed the answer for a host who does not know where the car is, and for such a host there is no trap to fear and no advantage to gain, only a fair coin.

## A cheque is an offer too

Selvin's 1975 sketch and Hall's dining room both contain a second kind of offer that the textbook version leaves out, and it obeys exactly the same arithmetic. In Selvin's letter the host offers to buy back the contestant's box for $100, then $200, then $500 and finally $1,000 (Selvin, 1975). In Tierney's account Hall offered money not to switch, starting at $3,000 and rising to $5,000, and explained afterwards that the rising offers were meant to make the contestant believe the car was behind the other door, when Hall knew it was behind the first (Tierney, 1991).

A cash offer is informative for the same reason a switch offer is. If the host makes it with probability c₁ when the contestant's door hides the car and c₀ when it does not, the probability that the contestant's door hides the car, given that the offer was made, is c₁ / (c₁ + 2c₀). A host who bids for the box only when it holds the keys reveals that it holds the keys; one who bids only when it is empty reveals that it is empty; and the only way to make the bid uninformative is to make it equally often in both cases, which means paying for empty boxes as often as he tries to recover full ones. Hall's description of his own tactics was that he exploited exactly this channel in the opposite direction, using the size of the offer to move the contestant's belief where he wanted it. The size of the cheque carries information only through the host's policy for writing cheques, and a contestant who does not know that policy cannot read the cheque, however large it is. Hall's advice at the end of the article, to take the money if he offers five thousand dollars not to open the door, is the advice of a man who knew his own policy and was telling his guest that the cheque was the only part of the game whose value did not depend on reading his mind.

## Two games with opposite answers

Whether that corner is a real risk depends on what the host is trying to do, and the cleanest way to ask is to treat the host as an opponent, as Nalebuff did in 1987 and Gill in 2011. In the textbook game the host must open a goat door and must offer the switch. The contestant who picks at random and switches with probability s wins with probability 1/3 + s/3, whatever the host does, so always switching guarantees 2/3, and the host can hold him to 2/3 by placing the car uniformly. The value of the game is two-thirds.

Now give the host the right he claimed in Tierney's article, to decide whether to offer at all. A contestant who picks at random and switches with probability s when offered wins

```
P(win) = (1 − a)/3 + a (1 − s)/3 + 2 b s / 3
```

The host, who chooses a and b, makes b = 0 and a = 1 and holds the contestant to (1 − s)/3. The contestant's best guarantee is to never switch, which wins 1/3 whatever the host does. The value of this game is one third, the same as having no offer at all. Jan Schuller analysed a malicious host with this freedom in 2012 and concluded that the classical answer is not a reliable guide on a real show (Schuller, 2012), and Mark Whitmeyer modelled a contestant who is uncertain whether the host is sympathetic or hostile as a Bayesian game in which the equilibrium depends on that prior (Whitmeyer, 2017).

The two games are the extremes of a family that makes the dependence explicit. Suppose the rules oblige the host to offer the switch in at least a share m of the games but leave him free to choose which. The host's best reply to any contestant who switches with positive probability is to offer every time the contestant holds the car (a = 1) and only as often as the rules force him otherwise. Solving the resulting linear programme gives

```
value(m) = max(1/3, m − 1/3)
```

which the script checks against a numerical solution at seven values of m. Below m = 2/3, the guaranteed win probability is 1/3, the contestant's maximin rule is never to switch, and the offer is worthless. Above it, the guarantee rises linearly to 2/3 at m = 1, and the maximin rule is to always switch: at m = 0.75 the value is 0.417 and at m = 0.9 it is 0.567. Figure 3 draws the kinked line. The threshold has a simple meaning. A host who must make offers in more than two games out of three can no longer confine them to the third of the games in which the contestant holds the car, and every offer beyond that third has to go to a contestant who picked a goat. Only then does the offer start to carry good news.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/monty-hall/part2/figures/fig3_offer_game.png)

*Figure 3. Win probability a contestant can guarantee against a host who must offer the switch in at least a given share of games but chooses which games. The dotted line is the worst case for a contestant who always switches.*

This is the formal content of what Hall told Tierney at the end of their afternoon: if the host is required to open a door every time and offer the switch, take it, and if he can choose, beware (Tierney, 1991). Tierney added that Hall had the option of not offering the switch on the real show and usually did not take it, and Hall's 1975 letter to Selvin said that contestants on *Let's Make a Deal* could not trade boxes once one had been opened. The textbook problem describes a game that, by its host's own account, was not played on his show. The game that was played, if the switch had ever been on offer, is the one with value one third, in which a contestant who reasons like vos Savant is exactly the kind of contestant a thrifty host would want to meet.

What a real host would do between the two extremes depends on what he is trying to achieve, and here the analysis can only offer a conjecture, labelled as such. A host who cared only about giving away as few cars as possible would play the trap. A host who cared only about television would offer the switch whenever it made for a tense moment. A host who cared about both, which is the natural description of a producer with a prize budget and a ratings target, would find that the cheapest tense moment is a contestant agonising over whether to give up a door that hides the car, and the conjecture is that such a host's policy would lie below the diagonal of Figure 2, offering more often to contestants who picked right. If that is so, the textbook advice is not merely unproven for a real show but biased against the contestant in the direction the host's incentives point. Testing the conjecture would need records of real offers and real first picks, which the published accounts of *Let's Make a Deal* do not provide.

## What each dial does, side by side

The four dials can now be compared on the three questions the article started with.

Can the dial make staying strictly better than switching? The tie-break cannot, since the probability that switching wins never falls below one half for any q. The host's knowledge cannot, since the worst case over knowledge and tie-break together is again one half. The placement can, for a contestant with a fixed first pick, as p = (3/5, 1/5, 1/5) shows. The offer policy can, all the way down to a sure loss.

Is there a rule that protects the contestant without knowing the dial? For the tie-break and the knowledge, always switching is optimal for every value, so it protects completely. For the placement, picking the first door at random and always switching guarantees exactly 2/3 whatever the placement. For the offer policy there is no such rule: the best guarantee against an unrestricted host is 1/3, achieved by ignoring the offer, and it rises above that only if the rules force the host to offer in more than two games out of three.

How much is it worth to learn the dial? The tie-break and the knowledge are worth nothing. The placement is worth 2/9 under a uniform prior, all of it upside. The offer policy is worth something between almost nothing and about three per cent of a car under the priors tried here, and the average hides that the cost of not knowing is concentrated in the hosts a contestant should fear.

The first two answers follow from one line of the general formula. With the offer compulsory (a = b = o = 1) and uniform placement, the bracket for the car behind door 2 is φ + (1 − φ)/2 and the bracket for door 1 is φq + (1 − φ)/2, and since q is at most one, the first is never smaller than the second. The likelihood ratio of the host's action is therefore at least one for every tie-break and every degree of knowledge, the posterior odds are at least the prior odds of one, and switching is at least as good as staying in every game. The offer policy is the only dial that can put a number smaller than one into that ratio without help from the placement, because b can be smaller than q·a.

The ranking is not the one the history suggests. The tie-break, the dial that absorbed most of the professional literature after 1991, is the one that matters least for the decision. The offer policy, the dial Hall pointed to in 1975 and acted out in 1991, is the only one that turns the advice around in a way the contestant cannot protect against, and the only one where the textbook game and the real show give opposite answers. In data-science terms, the tie-break is a calibration problem and the offer policy is a selection problem: the host chooses which games show the contestant an offer, and the contestant's conclusion depends on how that choice was made.

## Learning the host from past episodes

A contestant who cannot protect himself against the offer policy can try to learn it. Suppose that before playing he watches past episodes with the same host. In each one he sees whether the switch was offered and, at the end, whether the first pick was right. If the host's policy is stable, the episodes are data about a and b, and the decision rule follows from the formula above: switch when offered if 2b > a, and stay otherwise. With independent uniform priors on a and b, the posterior means after the episodes are (offers to right picks + 1)/(right picks + 2) and (offers to wrong picks + 1)/(wrong picks + 2), and the Bayesian contestant switches when twice the second exceeds the first. With no episodes at all the rule says switch, which is the textbook answer.

The script simulates 40,000 such contestants for each of five hosts and for numbers of past episodes from zero to a thousand, and records how often each one makes the right call and how often he wins the next game. Figure 4 shows the probability of the right call for the three hosts where the answer is in doubt.

Against the trap host, a = 1 and b = 0, learning is fast. The contestant who has seen no episodes switches and loses every game; the prior still recommends switching after one or two episodes, because so few right picks have been seen; and from five episodes on, the Bayesian contestant stays at least 95 per cent of the time. A host who never offers when the contestant is wrong cannot hide that for long.

Against the sparing host who leans towards right picks, a = 0.3 and b = 0.1, learning is slow. The right call is to stay, and the contestant only gets there by accumulating enough offers to see that they come disproportionately when the first pick was right. He makes the right call 46 per cent of the time after ten episodes, 70 per cent after fifty, 80 per cent after a hundred, 89 per cent after two hundred and 98 per cent after five hundred; on a finer grid, he stays above 95 per cent from about 340 episodes on. The difference in the win rate is small, 0.30 against 1/3, and it takes hundreds of games to detect.

The sparing host who leans the other way, a = 0.3 and b = 0.2, shows something less intuitive. The right call is to switch, and the contestant who has seen no episodes makes it, because the prior happens to point the right way. A handful of episodes then makes him worse. After ten episodes he switches only 74 per cent of the time, the probability of the right call bottoms out at 69 per cent around fifteen episodes, and it only climbs back above 95 per cent and stays there from about 400 episodes on. A small amount of data about a host who is close to the boundary is mostly noise, and noise is enough to overturn a default that was correct. The win rate follows the same dip: 0.358 after twenty episodes, against 0.367 for a contestant who never looked.

The dip is a general property of acting on estimates near a decision boundary, and it is common in practice. A pricing rule, a targeting threshold or an A/B decision that is switched on the strength of a few dozen observations is a rule that will often be switched the wrong way when the true effect is small, and the losses from those wrong switches can exceed the gains from the right ones until the sample is large. The Bayesian contestant here is not doing anything wrong: with a uniform prior on a and b, each of his decisions is the best one given what he has seen. What hurts him is that his prior does not encode what he actually knew before the first episode, namely that the textbook answer is the right one for a host whose offers do not depend on the contestant's luck. A prior concentrated near the diagonal a = b, which is what a contestant who believed the host to be roughly fair should hold, would have kept him switching until the data showed a clear asymmetry. The choice of prior is a statement about the host's incentives, and the dip is the price of stating them badly.

The same reasoning tells the contestant what to watch for. The decision depends on the ratio b/a, so episodes in which the switch is offered are the informative ones, and among those, the share in which the contestant had picked right. A host who offers rarely produces few informative episodes; the two sparing hosts above offer in a sixth and in under a quarter of games, which is why hundreds of episodes are needed. On the real show, where Hall said he usually did not offer the switch, a viewer would have had very little to learn from.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/monty-hall/part2/figures/fig4_learning.png)

*Figure 4. Probability that a Bayesian contestant makes the right call after watching a given number of past episodes, for three hosts. Blue: a contestant who sees every episode and its outcome. Orange: one who sees only the episodes in which the switch was offered.*

The orange curves in Figure 4 answer a question that matters for the next article. Suppose the contestant could only watch the episodes in which the switch was offered, as a broadcast might show only the dramatic moments, and could estimate only the share of offered games in which the first pick turned out to be right. That share is exactly the quantity the decision depends on, since switching wins when it is below one half, so this contestant is not misled. He learns less efficiently than the one who sees everything when offers are rare, and against the trap host he is better at the very start because his rule does not begin from the textbook answer, but his curves converge to the same place. The cost of the narrower view shows most clearly against the standard host, who offers every time: the contestant who sees every episode never leaves the right answer, because each episode confirms that offers do not depend on luck, while the one who sees only offered episodes and their outcomes makes the right call 67 per cent of the time after one episode, 92 per cent after ten and 96 per cent after twenty. Knowing how often the host declines to offer is itself information, and a broadcast that cuts the games without an offer throws it away. Selection of what is observed is harmless here because what is selected, the offered games, is exactly the population the decision concerns. The third article in this series is about the cases where that is not true: where the games one sees have been selected by a mechanism related to their outcome, and the selected sample answers a different question from the one being asked.

## What the dials mean for data scientists

The Monty Hall problem is small enough that every component of the data-generating process can be written down and turned separately, and that makes it a clean laboratory for a habit that is hard to practise on real models. Before arguing about whether an assumption is right, it is worth asking what depends on it.

Some assumptions are like the tie-break. Getting them wrong changes a reported probability, sometimes a lot, and changes no decision. A model with a mis-specified tie-break gives a contestant badly calibrated beliefs about the game after door 3 opens and exactly the right action. Effort spent refining such an assumption improves the report and not the outcome, and when the decision is the product, it is effort spent in the wrong place. A credit model whose default probabilities are off by a few points for applicants far from the approval cut-off is the same case: the miscalibration matters for loss provisioning, where the probability itself is used, and not for the approve-or-decline decision, where only the side of the threshold matters.

Some assumptions are like the placement. They can change the decision, but there is a way to act that is robust to them, here randomising the first pick, and learning them is pure upside. The sensible order is to adopt the robust action first and then learn.

And some are like the offer policy. They describe a choice made by someone who knows more than the analyst, after seeing what the analyst did, and they can make the default action the worst one available. No amount of thinking about the observed outcome substitutes for knowing how that choice was made, and data help only if they come from the same chooser and are read with the selection in mind. In the Monty Hall problem that chooser is a game-show host. In data work it is a ranking algorithm that decides which items a user is shown, a clinician who decides which patients are tested, a lender who decides which applicants receive a loan and therefore have a repayment record, or a platform that decides which experiments to report. The host's offer is the simplest example of a selection mechanism, and the next article in this series takes that view of the whole problem: the host as a process that decides which part of the truth the contestant gets to see, and the Monty Hall problem as a case of data that are missing not at random.

---

The code for every number and figure in this article is in the series repository: [github.com/DiogoRibeiro7/Medium-Blog/tree/main/monty-hall/part2](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/monty-hall/part2). `protocol_numbers.py` enumerates the general host exactly, checks the closed forms, computes the values of information, solves the game with a minimum offer rate and simulates the learning contestants; `figures.py` draws the four figures.

## References

Gill, R. D. (2011). The Monty Hall problem is not a probability puzzle (it's a challenge in mathematical modelling). *Statistica Neerlandica* 65(1), 58–71. https://arxiv.org/abs/1002.0651

Howard, R. A. (1966). Information value theory. *IEEE Transactions on Systems Science and Cybernetics* 2(1), 22–26.

Morgan, J. P., Chaganty, N. R., Dahiya, R. C. and Doviak, M. J. (1991). Let's make a deal: The player's dilemma. *The American Statistician* 45(4), 284–287.

Nalebuff, B. (1987). Puzzles: Choose a curtain, duel-ity, two point conversions, and more. *Journal of Economic Perspectives* 1(2), 157–163.

Raiffa, H. and Schlaifer, R. (1961). *Applied Statistical Decision Theory*. Division of Research, Graduate School of Business Administration, Harvard University, Boston.

Rosenthal, J. S. (2008). Monty Hall, Monty Fall, Monty Crawl. *Math Horizons* 16(1), 5–7. https://www.probability.ca/jeff/writing/montyfall.pdf

Schuller, J. C. (2012). The malicious host: a minimax solution of the Monty Hall problem. *Journal of Applied Statistics* 39(1), 215–221.

Selvin, S. (1975). On the Monty Hall problem (letter to the editor). *The American Statistician* 29(3), 134.

Tierney, J. (1991). Behind Monty Hall's doors: Puzzle, debate and answer? *The New York Times*, 21 July 1991, section 1, p. 1. https://www.nytimes.com/1991/07/21/us/behind-monty-hall-s-doors-puzzle-debate-and-answer.html

Whitmeyer, M. (2017). The Monty Hall problem as a Bayesian game. *Games* 8(3), 31. https://doi.org/10.3390/g8030031
