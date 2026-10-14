# Feedback Coder

**LLM-assisted qualitative coding of user feedback – with every answer checked before it is trusted.**

Product teams collect hundreds of app reviews, survey answers and support tickets. Reading and
categorising them by hand ("coding" them, in research terms) is slow. An LLM can do it in seconds, but an
LLM is also confident when it is wrong. This project shows how to use one *responsibly*:

1. a **codebook** with clear rules, shared by the model and the human coder,
2. a **fixed JSON format**, validated with Pydantic, with automatic retries,
3. **verification**: every code needs a quote copied from the comment, and quotes that do not exist are rejected,
4. **evaluation** against human labels with precision, recall, F1 and **Cohen's kappa**,
5. a **keyword baseline**, so the result has something honest to be compared with,
6. an interactive **HTML report** with themes, evidence, disagreements and a human-review queue.

**Live report:** https://ngnmsv.github.io/feedback-coder/

## Results

60 comments, 8 codes, compared with human labels made with the same codebook.

| Run | Micro-F1 | Macro-F1 | Mean kappa | Same code set |
|---|---|---|---|---|
| Keyword baseline (no AI) | 0.68 | 0.70 | 0.66 | 48% |
| LLM (Claude Haiku 5.5) | **0.96** | **0.97** | **0.96** | 90% |

The baseline is precise (0.89) but misses almost half of the codes (recall 0.56): it cannot see a feature
request without the words "please add", or praise without "love" or "great". That gap is what the LLM is for.

The LLM closes it: precision and recall both 0.96. All its quotes passed the evidence check (no invented
quotes, no unknown codes, no retries needed), and 7 comments were sent to human review for low confidence.
Its 6 disagreements with the human coder are mostly about `praise` (e.g. "works perfectly with Volksbank" –
is that praise or only a bank-sync remark?), which points at a codebook rule to sharpen rather than a model error.

## Quick start

```bash
git clone https://github.com/NgnMsv/feedback-coder.git
cd feedback-coder
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m feedback_coder run --provider offline        # no key needed, takes a second
open output/report.html                                 # Windows: start output\report.html
pytest                                                  # 42 tests
```

### With a real model

```bash
cp .env.example .env            # then paste your key into .env
python -m feedback_coder run --provider anthropic      # or: --provider openai
python -m feedback_coder run --provider anthropic --publish docs/index.html   # update the live report
```

A full run is 6 requests (10 comments each) and costs a few cents with a small model. Every answer is
saved in `runs/`, so running again – or presenting the result on another computer without a key – uses
the saved answers:

```bash
python -m feedback_coder run --provider anthropic --cached
```

Other commands: `python -m feedback_coder prompt` prints the exact prompt sent to the model.

## How it works

```
codebook.yaml ─┐
               ├─► prompt ─► model ─► JSON ─► schema check ─► verification ─► evaluation ─► report.html
feedback.csv ──┘                       ▲         │ invalid                        ▲
                                       └─ retry ◄┘                     gold_labels.csv (human)
```

| File | What it does |
|---|---|
| `codebook.yaml` | The 8 codes: definition, include/exclude rules, examples |
| `data/feedback.csv` | 60 fictional comments about a budgeting app |
| `data/gold_labels.csv` | Human coding of the same comments |
| `src/feedback_coder/prompts.py` | Builds the prompt from the codebook |
| `src/feedback_coder/providers/` | OpenAI and Claude over plain HTTP, plus the offline keyword baseline |
| `src/feedback_coder/schema.py` | The JSON shape the model must return (Pydantic) |
| `src/feedback_coder/verify.py` | Rejects unknown codes and invented quotes, flags low confidence |
| `src/feedback_coder/pipeline.py` | Batching, retries with the error message, cache |
| `src/feedback_coder/evaluate.py` | Precision, recall, F1, Cohen's kappa – by hand, no library |
| `src/feedback_coder/report.py` | Self-contained HTML report with SVG charts |

`LEARNING.md` walks through the project lesson by lesson. `PRESENTING.md` is a guide to presenting it.

## Limitations

- The 60 comments are fictional, written for this demo, and coded by one person. A real study needs real
  data and at least two human coders, so that human–human agreement can be measured too.
- The codebook is fixed in advance (deductive coding). Grounded theory starts the other way round, by
  discovering codes in the data; see the ideas at the end of `LEARNING.md`.
- Sending real user feedback to an API is a privacy decision. Remove personal data first, or use a local model.

## License

MIT
