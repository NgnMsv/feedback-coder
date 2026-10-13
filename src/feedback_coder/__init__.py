"""feedback-coder: LLM-assisted qualitative coding of user feedback, with checks.

The pipeline, in order (each step is one module):

    data.py      load comments and the human (gold) labels
    codebook.py  load the categories and their rules
    prompts.py   turn the codebook + a batch of comments into a prompt
    providers/   send the prompt to a model (OpenAI, Claude) or run the offline baseline
    schema.py    the exact JSON shape the model must return
    verify.py    check every answer: valid codes, real quotes, confidence
    pipeline.py  batching, retries, caching – ties the steps together
    evaluate.py  compare the result with the human labels (F1, Cohen's kappa)
    themes.py    turn coded comments into themes and counts
    report.py    write the interactive HTML report
    cli.py       the command line: `python -m feedback_coder run ...`
"""

__version__ = "1.0.0"
