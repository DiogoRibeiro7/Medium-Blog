# Codebook: disciplinary interventions by the Chair

One label per Chair speech: **1** if it contains a disciplinary intervention, **0** if it
does not, **u** if the text and its context do not allow a decision.

Label blind. Do not open `design.csv` or the stage 1 results before you have finished:
they show which speeches the dictionary tagged, and the labels exist to check the
dictionary.

## The construct

A disciplinary intervention is the Chair acting to **restrain the conduct** of a Member
or of the House. It is about behaviour, not about time and not about procedure.

## Label 1 when the Chair speech contains any of these

- A call for order or quiet that responds to noise, shouting, heckling or interruption.
  A bare "Order." or "Order, order." with nothing else counts: the Chair does not call
  order in a quiet House.
- "Order" followed by business, when the speech before shows disorder (an
  `[Interruption.]`, a "Hon. Members:" line, a Member being shouted down) and the call
  answers it.
- An instruction to a Member to sit down, to resume their seat or to stop speaking
  because of how they are behaving.
- A request to withdraw a remark; a ruling that language is unparliamentary; a rebuke of a
  Member's language, tone, gestures or behaviour, however politely phrased.
- Naming a Member, ordering a Member to leave the Chamber, or suspending the sitting
  because of disorder.
- A point of order ruled out **with** a rebuke ("That is not a point of order, and the hon.
  Gentleman knows it").

## Label 0 when the Chair speech only does these

- Calls a Member, announces business, puts a Question, reads a result or a procedure.
- Says "Order." as a transition before announcing business or calling the next speaker,
  and nothing in the speech or in the one before it shows disorder ("Order. Statement,
  the Secretary of State.").
- Rules on a point of order without a rebuke, including a plain "That is not a point of
  order".
- Asks Members to be brief, to keep to time or to let others in. That is time-keeping, a
  different construct; label it 0 here unless the same speech also restrains behaviour.
- Thanks, welcomes, informs, or makes a statement.
- Suspends the sitting for a reason other than disorder (a division, a technical fault,
  safe exit from the Chamber).

## Unsure

Use **u** sparingly, for speeches where neither the text nor the context settles it.
Unsure items are left out of the estimates, which assumes they are not systematically
different from the labelled items in their stratum. Add a note (`n` in the tool) when a
case needed a judgement the rules above do not cover; the notes are how the codebook
gets revised for the next sample, never for this one.

## The tool

```bash
hansard-drift label --annotator yourname
```

It shows the speech before, the Chair's speech and the speech after, saves after every
answer and resumes where you stopped. With two annotators, `hansard-drift agreement`
reports Cohen's kappa and writes the disagreements; resolve them into `labels/labels.csv`
(`id,discipline,unsure,note`), which the estimates then use.
