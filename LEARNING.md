# Learning guide

This guide takes you through the project in the order the data flows. Each lesson has:

- **Read** – the file(s) to open, top to bottom
- **Key ideas** – what you must be able to explain in your own words
- **Try it** – a small change to make and run
- **Check yourself** – questions an interviewer could ask

Plan about 1–2 hours per lesson. Do the lessons in order: each one builds on the last.
By the end you should be able to rebuild any part of the project from a blank file.

---

## Lesson 0 – Run it

**Do:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m feedback_coder run --provider offline
pytest
```
Open `output/report.html`. Click through the filters, hover the bars, switch to dark mode.

**Key ideas**
- `pip install -e .` installs the package in *editable* mode: you change the code, the next run uses it.
- `python -m feedback_coder` runs `src/feedback_coder/__main__.py`, which calls `cli.main()`.
- The `src/` layout keeps the package separate from tests and data.

**Check yourself**
- What three files does a run write, and where?
- What does the offline provider do differently from the other two?

---

## Lesson 1 – The problem and the codebook

**Read:** `README.md`, `codebook.yaml`, `data/feedback.csv`, `data/gold_labels.csv`, `src/feedback_coder/data.py`, `codebook.py`

**Key ideas**
- *Qualitative coding* = assigning categories ("codes") to pieces of text so you can count and compare them.
- It is **multi-label**: "Got charged twice and support hasn't replied" is both `pricing` and `support`.
- A **codebook** makes coding repeatable. Labels alone are ambiguous; *include* and *exclude* rules decide
  the borderline cases. Example: "Sync with PayPal would be a game changer" – is it `bank_sync` or
  `feature_request`? The exclude rule under `bank_sync` decides it: a *new* bank is a feature request.
- The **gold labels** are the human reference. Everything we measure later is "agreement with this human".

**Try it – this one matters most:** You are the human coder now. Go through `data/gold_labels.csv` and
check every label against the codebook. Change any you disagree with; they must be *your* labels, because
you will defend them in an interview. Note the three hardest cases. Those are your best interview examples.

**Check yourself**
- Why can't you just give the model the eight labels without definitions?
- Why is "The app is okay." coded with no code at all?
- What is the difference between a code (topic) and the sentiment?

---

## Lesson 2 – Prompt design

**Read:** `src/feedback_coder/prompts.py`, then run `python -m feedback_coder prompt`

**Key ideas**
- **System message** = role + codebook + rules (same for every batch). **User message** = the data (one batch).
- The prompt asks for **evidence**: the exact words that justify each code. This makes the model show its
  work, and gives us something we can check automatically (Lesson 5).
- Comments are sent as JSON **with ids**, and must come back with the same ids. Without ids you could
  never be sure which answer belongs to which comment.
- **Batching** (10 comments per request): fewer requests, lower cost, but too large a batch makes the model
  sloppier. 10 is a reasonable middle.

**Try it**
- Remove the `Examples:` line from `system_prompt()` and look at the printed prompt. What information does
  the model lose? (Later, with an API key, run both versions and compare the F1.)

**Check yourself**
- Why `temperature = 0`?
- What is "few-shot prompting", and where does this project use it?

---

## Lesson 3 – Calling an LLM API by hand

**Read:** `providers/base.py` (`post_json`), `providers/openai_client.py`, `providers/anthropic_client.py`

**Key ideas**
- An LLM API call is just an **HTTP POST**: a URL, headers (your API key) and a JSON body. No SDK needed;
  this project uses only `urllib` from the standard library, so nothing is hidden.
- OpenAI puts the system prompt inside `messages`; Anthropic has a separate `system` field and returns a
  list of content blocks. Same idea, slightly different shapes. That is why each has its own small client.
- `response_format: json_object` (OpenAI) forces syntactically valid JSON, but it does **not** guarantee the
  right fields. That is still our job (Lesson 4).
- Errors: **429** (rate limit) and **5xx** (server) are retried with exponential back-off (1 s, 2 s, 4 s).
  A 401 (bad key) is not retried: retrying will not fix it.
- The **API key** lives in `.env`, which is in `.gitignore`. Never commit a key.

**Try it**
- Get an API key (Anthropic console or OpenAI platform), put it in `.env`, and run
  `python -m feedback_coder run --provider anthropic`. Watch the progress lines.
- Read `tests/test_providers.py`: how do the tests check the HTTP request *without* calling the API?

**Check yourself**
- What would happen if you put the key in the code and pushed it to GitHub?
- Why is a 429 retried but a 401 not?

---

## Lesson 4 – Structured output and retries

**Read:** `schema.py`, then `parse_answer()` and the retry loop in `pipeline.py`

**Key ideas**
- An LLM returns **text**. Pydantic turns it into a typed object, or a precise error such as
  "confidence: Input should be less than or equal to 1".
- **Retry with feedback**: on an invalid answer, the *exact* validation error is sent back to the model
  ("your previous answer could not be used because…"). Models fix their own mistakes well when told what they were.
- After 3 failed attempts the batch is **not** dropped silently: every comment in it is flagged
  `invalid_output` for human review.
- `parse_answer` tolerates a ```` ```json ```` fence, a common habit of chat models.

**Try it**
- In `tests/test_pipeline.py`, read `test_retry_sends_the_error_back_and_then_succeeds`. Change the fake's
  first answer to valid JSON with `"confidence": 7`. Does the test still pass? Why?

**Check yourself**
- Why validate even when the API promises JSON?
- Where does the retry count show up in the report?

---

## Lesson 5 – Verification: never trust, always check

**Read:** `verify.py` (the most important file), `tests/test_verify.py`

**Key ideas**
- Five checks: `unknown_code`, `evidence_missing`, `low_confidence`, `missing_result`, `extra_result`.
- **Evidence check** = a cheap, automatic **hallucination detector**. If the "quote" is not in the comment,
  the model invented or paraphrased it, and the code is dropped.
- `normalize()` makes the check fair: case, extra spaces, curly vs straight quotes and trailing full stops
  do not count as a fake quote.
- **Drop vs. flag**: an invented quote is *dropped* (we cannot trust that code); low confidence is *kept but
  flagged* (probably right, a human should glance at it). Every decision is visible in the report.
- This is **human-in-the-loop**: the model does the bulk of the work, and the checks decide what a human
  must look at.

**Try it**
- Lower `LOW_CONFIDENCE` to 0.3 or raise it to 0.9, rerun with your cached LLM run (`--cached`) and
  compare the "Sent back to a human" count. What is the trade-off?

**Check yourself**
- Give an example of a hallucination this project catches, and one it cannot catch.
  (Hint: a *real* quote attached to the *wrong* code is not caught by the evidence check, only by evaluation.)

---

## Lesson 6 – Evaluation: is it actually right?

**Read:** `evaluate.py`, `tests/test_evaluate.py`

**Key ideas**
- Each code is a yes/no question over all comments → a 2×2 table of TP, FP, FN, TN.
- **Precision** = TP / (TP + FP): when it says yes, is it right? **Recall** = TP / (TP + FN): does it find them all?
- **F1** balances both. **Micro** F1 pools all codes (frequent codes count more); **macro** F1 averages the
  codes (rare codes count equally).
- **Cohen's kappa** corrects agreement for chance:
  `kappa = (p_observed − p_expected) / (1 − p_expected)`. It is the standard measure for comparing two
  coders in qualitative research, which is exactly what the model is: a second coder.

**Try it – by hand, on paper:** For `praise` in the baseline run, the report says precision 0.60 and recall
0.21 with 14 human-coded comments. Work out TP, FP and FN. (Answer: TP 3, FP 2, FN 11.) Then compute the
kappa yourself with TN = 60 − 3 − 2 − 11 = 44 and check it against the table (0.22).

**Check yourself**
- Why is plain accuracy a bad measure here? (Hint: most comments do *not* have a given code.)
- The baseline's precision is high but its recall is low. What does that tell you about keyword rules?

---

## Lesson 7 – Baselines and a mistake to avoid: data leakage

**Read:** `providers/offline.py`, the comment at the top of `codebook.yaml`

**Key ideas**
- A number without a comparison means little. "The LLM reaches 0.9" only matters next to "a simple
  keyword list reaches 0.68".
- **Data leakage – a real story from building this project:** the first keyword lists were written while
  looking at the 60 test comments. The baseline scored **0.92**, nearly perfect, because the keywords
  were copied from the answers. Rewritten from the code *definitions* only, it scores **0.68**. Rule:
  never tune anything on the data you evaluate on.
- The baseline's typical errors are instructive: "Not slow anymore, thanks!" contains "slow" (it cannot read
  negation), and "would make this perfect" is a feature request with no request keyword in it.

**Try it**
- Look at the "Where model and human disagree" list for the baseline. Sort the errors into: negation,
  missing keyword, wrong keyword. Then do the same for your LLM run. What kinds of errors does the LLM still make?

**Check yourself**
- Explain data leakage in one sentence, with this project as the example.

---

## Lesson 8 – Cost, caching and reproducibility

**Read:** `Cache` and `cache_key()` in `pipeline.py`, `runs/`

**Key ideas**
- The cache key is a **hash of the prompt** (provider, model, codebook and batch). Change anything that
  changes the prompt → new key → a new API call. Change nothing → free, identical result.
- Commit `runs/*.jsonl`: anyone can reproduce your exact numbers with `--cached`, with no key and no cost.
  That is how you present a real model run on any computer.
- Rough cost: the system prompt is about 700 words; 6 batches with a small model cost a few cents.

**Check yourself**
- Why hash the prompt instead of using the batch number as the cache key?

---

## Lesson 9 – Themes and the report

**Read:** `themes.py`, `report.py`

**Key ideas**
- Themes = counts per code, sentiment per code, codes that appear together, and the highest-confidence
  quotes as evidence. This is what a product manager reads.
- The report is one **self-contained HTML file**: SVG charts drawn in Python, a few lines of JavaScript for
  filters and tooltips, no CDN, no server. It works offline and on GitHub Pages.
- **Security**: every value from the data goes through `html.escape`. A comment containing `<script>`
  is shown as text, never executed (see `test_comment_html_is_escaped`).

**Try it**
- Add a sixth KPI tile: "average confidence". You need to change `render()` only.

---

## Lesson 10 – Tests and CI

**Read:** `tests/`, `.github/workflows/ci.yml`

**Key ideas**
- 42 tests, none of which calls a real API: a **fake coder** returns prepared answers, and `mock.patch`
  replaces `urllib.request.urlopen`. Tests are fast, free and deterministic.
- The tests cover the risky parts: verification rules, the retry loop, the kappa formula (against a textbook
  example), HTTP request shapes, HTML escaping, and one end-to-end run.
- GitHub Actions runs lint, the tests and an end-to-end offline run on every push.

**Try it**
- Write one new test: a comment where the model returns the same quote for two different codes.
  What *should* happen? Write the test first, then check the code.

---

## Lesson 11 – Make it yours and publish

1. Review the gold labels (Lesson 1) and commit your changes.
2. Run with a real model: `python -m feedback_coder run --provider anthropic --publish docs/index.html`
3. Fill in the LLM row of the results table in `README.md` from `output/metrics.json`.
4. Commit `runs/`, `docs/index.html` and the README, then push. GitHub Pages serves the new report.
5. Update the "Results" part of the portfolio case study with the real LLM numbers.

## Ideas to extend it (good interview talking points)

- **Two models as two coders**: run Claude and OpenAI, compute kappa *between them*, and send only
  disagreements to a human.
- **Self-consistency**: run the same model 3 times and keep codes that appear in at least 2 runs.
- **Inductive coding (closer to grounded theory)**: first ask the model to *propose* codes from a sample,
  merge them into a codebook by hand, then code deductively as here.
- **Native structured output / tool use**: let the API enforce the schema instead of validating afterwards,
  and compare the retry counts.
- **A local open-source model** (for example via Ollama) for data that must not leave the computer.
- **German comments**: add some and check whether the evidence check still works across languages.
