# Presenting this project

## The 30-second version

> "Product teams get hundreds of user comments and categorising them by hand is slow. I built a pipeline
> that uses an LLM to do this coding, but never trusts it blindly. The model has to follow a codebook,
> answer in a fixed JSON format and quote the exact words behind every category. My code then checks
> those quotes, sends unclear cases to a human, and measures agreement with human labels using F1 and
> Cohen's kappa, compared against a simple keyword baseline. The result is an interactive report a
> product team can actually use."

Adjust the wording until it sounds like you. Say it out loud a few times.

## The 3-minute demo (with the live report open)

1. **The problem (20 s).** Scroll to "All comments". "These are 60 comments about a budgeting app. Someone
   has to read and sort them, every week."
2. **The themes (30 s).** Go back to the top: KPIs, then the bar chart. "Bank sync and bugs come up most.
   Every theme has quotes, so a product manager can see the evidence, not just a number."
3. **Trust (60 s): the important part.**
   - "Every code needs a quote copied from the comment. If the quote does not exist, the model made it up,
     and the code is dropped." Show a highlighted quote in the table.
   - Show "Sent back to a human": "Low confidence or failed checks are never used silently."
4. **Measurement (45 s).** Show "Does it agree with a human coder?". "I coded all 60 comments myself with
   the same codebook. The model reaches kappa X; the keyword baseline only 0.66. Here is where they
   disagree, and each disagreement is either a model error or an unclear rule in my codebook."
5. **Engineering (25 s).** Open GitHub: "Plain HTTP to OpenAI and Claude, Pydantic validation with
   retries, a cache so runs are free and reproducible, 42 tests, CI on every push."

## Questions you will probably get, and good answers

**"Why not just paste the comments into ChatGPT?"**
Because then you cannot check or repeat it. Here every answer is validated, every code has evidence that
is checked automatically, the result is measured against human labels, and the same input always gives
the same output, from the cache.

**"How do you know the model is right?"**
Three layers: the schema check (correct format), the evidence check (the quote really exists), and the
evaluation against human labels (F1 and kappa per code). Plus a review queue for anything uncertain.

**"What hallucinations can it NOT catch?"**
A real quote attached to the wrong code. The evidence check passes, and only the evaluation against human
labels reveals it. That is why measuring against humans matters.

**"Why Cohen's kappa and not accuracy?"**
Most comments do not have any given code, so a model that always says "no" would score high accuracy.
Kappa corrects for chance agreement, and it is the standard way to compare two coders in qualitative research.

**"What does it cost?"**
Six requests for 60 comments, a few cents with a small model. Answers are cached, so re-running is free.

**"What about privacy?"**
Real feedback can contain personal data. I would remove it before sending, or use a local open-source
model. The code already separates the model client from the rest, so swapping it is one new small file.

**"What would you do next?"**
Use two models as two independent coders and only send their disagreements to a human, or let the model
first *propose* codes from the data (inductive coding) before coding with a fixed codebook.

**"What was hard?"**
A good honest answer: "My first keyword baseline scored 0.92. I realised I had written its keywords while
looking at the test comments: data leakage. Rewritten from the definitions only, it scores 0.68. That
taught me to never tune anything on the evaluation data."

## Connecting it to your research (adjust to what is true for you)

> "In my research I use LLMs like ChatGPT, LLaMA and Phi-3 to analyse user feedback. This project is a
> small, public version of that idea, which I built to show the engineering side: validation, verification
> and measurement."

## Before an interview

- [ ] Run it once yourself from a fresh clone (Lesson 0)
- [ ] Know your own three hardest gold-label decisions (Lesson 1)
- [ ] Be able to compute precision, recall and kappa on paper (Lesson 6)
- [ ] Have the live report open in a browser tab
- [ ] Know the real numbers of your LLM run by heart
