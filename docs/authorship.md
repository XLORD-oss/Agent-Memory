# Handing over first authorship — how to make it real, not decorative

You have decided the collaborating group's student takes first author and
you take second, because a senior group's paper is worth more to you than a
solo byline. Good trade. But it raises the exact question you asked: **first
authorship has to be earned by the work, and you cannot "offer" it the way you
offer a coffee.** If the first author did not do first-author work, three
things go wrong: the PI will not accept it (good PIs police this), the student
cannot defend the paper at a talk, and any reviewer who reads the git history
sees a mismatch.

So the mechanism is not "offer first authorship". It is: **transfer ownership
of the parts of the study that first authors are supposed to own, and let the
byline follow.**

## 1. What first authors are supposed to own

By the norms every ML venue actually uses (ICMJE-style substantial
contribution + CRediT roles), the first author is the person who can answer
"why did you do it this way?" for the paper's *central experiment* — not the
tooling around it. Concretely, four things:

| Component | Who owns it now | What it means to own it |
|---|---|---|
| **A. The central experiment** — the H4 fine-tuning study and the cross-eval matrix | nobody yet; only the design and the data exporter exist | choose base model, LoRA config, data, run the 12 adapters, own every failure, decide what "the result" is |
| **B. The baselines** — RAG, rolling summary, MemGPT-style | a reference `rolling_summary` stub exists | implement, tune fairly (same token budget), defend them against "strawman" |
| **C. The measurement** — item set, human labels, flip detector validity, confidence metric | 8 trivia items and a regex | pick the item set, run the labelling, report κ, choose the calibration measure |
| **D. The analysis and the story** — stats, figures, the argument of the paper | scaffolding (`aggregate.py`) | pre-registration, paired tests, the figure that carries the paper, the first draft of Results/Discussion |

You own the framework, the benchmark harness, the pilot, and the hypothesis.
That is a *second-author-plus* contribution already: substantial software,
conceptualisation, and the first evidence. It does not diminish by handing
A–D away; it is what makes handing them away possible.

**Rule of thumb:** whoever owns A owns first authorship. B–D distribute the
rest. If the student ends up owning A and D, first authorship is not a gift,
it is a description.

## 2. How to hand over without it becoming "do my experiments"

The failure mode is you writing the plan, them executing it, and everyone
pretending that is ownership. Avoid it structurally:

1. **Freeze what is yours, open what is theirs.** The framework, harness and
   pilot are done and cited as-is. A–D are explicitly *undecided* in the
   proposal: "we have a design and a data tool; the study itself is to be
   designed by the group." Do not arrive with LoRA hyperparameters.
2. **The review meeting is where they take it.** Ask them to break the H4
   design (`training.md`). Every change they make becomes theirs. Write the
   pre-registration *with* the student as the drafting author.
3. **Separate, attributable code.** Their contributions live in their own
   modules with their name in the docstring and git history:
   `benchmarks/baselines/<name>.py` for baselines (the arm registry makes
   each baseline a drop-in, no edits to the core runner), `training/` for the
   fine-tuning configs and scripts, `analysis/` for the notebooks. Git blame
   should show who did what without anyone having to argue it.
4. **Decision log.** A short `docs/decisions.md` (or the PR descriptions)
   recording each design choice and who made it. Painless at the time,
   decisive if authorship is ever questioned.
5. **You stay the integrator, not the author of their parts.** You review
   their PRs, keep CI green, keep the harness stable under them. That is a
   real, visible, second-author role — and it keeps you close enough to
   the science to co-write the Discussion.

## 3. What if they contribute *less* than first-author work?

It happens — the student is overloaded, the PI's interest fades, the runs get
done but the thinking does not. Set the rule at the review meeting so nobody
has to renegotiate under pressure:

> "Authorship order follows the CRediT table we fill in today. We revisit it
> once, at the point the main results are in, against what actually happened.
> If the group led the central experiment and the analysis, [student] is
> first. If it turned out that I did, we go co-first or I go first — and
> either way the PI is last."

Written into the same email thread as the pre-registration. A serious PI
will be relieved you said it; it is the conversation they usually have to
start.

**Co-first authorship** (†) is the honest middle and increasingly standard in
ML: both names first, equal-contribution footnote, alphabetical or by
agreement. If the split is genuinely A+D theirs / framework+pilot+B yours,
co-first is more accurate than either ordering, and it costs nobody anything.

## 4. What this looks like in the proposal

The one-page ask changes in one place. Instead of "I do the engineering, run
everything, and write the first draft":

> **Roles.** The framework, harness and pilot are done and provided as-is.
> The group's student leads the central study (H4 fine-tuning and cross-eval),
> the baselines, and the analysis, and drafts the paper; I integrate,
> maintain the harness, run the inference campaign with them, and co-write.
> First author: the student; second: me; last: [PI] — settled today in the
> CRediT table, revisited once against actual contribution when results
> are in.

That sentence is what makes a PI say yes: it offers their student a
first-author paper *with the hard engineering already done* — the best kind
of first-author project — while making clear the student must actually do
the science.

## 5. The CRediT table, rewritten for this split

| Role | Student (1st) | You (2nd) | PI (last) |
|---|---|---|---|
| Conceptualization | ○ | ● | ○ |
| Methodology (study design, pre-registration) | ● | ○ | ● |
| Software — framework, harness, pilot | | ● | |
| Software — baselines, training, analysis code | ● | ○ | |
| Investigation — inference campaign | ○ | ● | |
| Investigation — fine-tuning study (H4) | ● | ○ | |
| Data curation (item set, labels) | ● | | |
| Formal analysis | ● | ○ | ○ |
| Visualization | ● | ○ | |
| Writing – original draft | ● | ○ | |
| Writing – review & editing | ○ | ● | ● |
| Supervision | | | ● |
| Resources (compute) | | | ● |

● lead ○ contributing. Read down the Student column: it is a first author's
column. Read down yours: it is a strong second — and the only person in the
table who can maintain the artefact after the paper, which is what people
cite.

## 6. What you keep regardless

* The repository, its name, the MIT licence, and the `CITATION.cff` — the
  *software* citation is yours independent of the paper's author order.
* The framework's design documents (`unified.md`, `map.md`, `claim.md`) with
  their history.
* The next paper. A first author who owns A–D on this one leaves you the
  natural first author on the follow-up (paraphrased-replay arm, RLHF-vs-SFT,
  the state-estimation framing) — and now with a senior co-author who knows
  the work.

## 7. Red flags — when *not* to hand it over

* The PI wants first authorship for a student who will "run the scripts".
  That is not ownership; offer second and hold the line.
* Nobody in the group will commit to the design meeting. Without design
  ownership there is no first-author claim, whatever the compute.
* They want the repo transferred or relicensed. The paper is shared; the
  tool is not part of the deal.
