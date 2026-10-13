"""Step 9 – write one self-contained HTML report (no server, no CDN).

The report has five parts, in the order you would present them:
  1. Summary numbers   – how many comments, how much agreement, how many need review
  2. Themes            – what users talk about, with quotes as evidence
  3. Agreement         – per-code precision, recall, F1 and kappa vs the human labels
                         (and vs the keyword baseline, if given)
  4. Needs review      – comments the checks sent back to a human, and why
  5. All comments      – filterable table with the evidence highlighted

Charts are drawn as inline SVG by the Python below, so the file works offline
and can be opened by double-clicking it. Every value from the data goes
through `esc()` (HTML escaping), so a comment containing <script> is shown as
text and never executed.
"""

from __future__ import annotations

import html
import json

from .codebook import Codebook
from .evaluate import Evaluation
from .pipeline import RunResult
from .themes import Theme, co_occurrence
from .verify import normalize


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def highlight(text: str, quotes: list[str]) -> str:
    """Escape `text` and wrap each evidence quote in <mark>."""
    spans: list[tuple[int, int]] = []
    lower = text.lower().replace("’", "'")
    for q in quotes:
        target = normalize(q)
        if not target:
            continue
        i = lower.find(target)
        if i >= 0:
            spans.append((i, i + len(target)))
    spans.sort()
    out, pos = [], 0
    for start, end in spans:
        if start < pos:  # overlapping quote: skip
            continue
        out.append(esc(text[pos:start]))
        out.append(f"<mark>{esc(text[start:end])}</mark>")
        pos = end
    out.append(esc(text[pos:]))
    return "".join(out)


# ---------------------------------------------------------------- charts (SVG)

def theme_bars(themes: list[Theme], total: int) -> str:
    """Horizontal bars: number of comments per theme (one series, so no legend)."""
    row_h, label_w, bar_w, w = 34, 170, 330, 700
    h = row_h * len(themes) + 10
    top = max((t.count for t in themes), default=1) or 1
    rows = []
    for i, t in enumerate(themes):
        y = i * row_h + 6
        bw = max(t.count / top * bar_w, 2 if t.count else 0)
        tip = f"{t.label}: {t.count} of {total} comments ({pct(t.share)}), {pct(t.negative_share)} negative"
        rows.append(
            f'<g class="hit" data-tip="{esc(tip)}" tabindex="0">'
            f'<rect x="0" y="{y - 4}" width="{w}" height="{row_h}" fill="transparent"/>'
            f'<text x="{label_w - 12}" y="{y + 15}" text-anchor="end" class="axis">{esc(t.label)}</text>'
            f'<rect x="{label_w}" y="{y + 2}" width="{bw:.1f}" height="18" rx="4" class="bar s1"/>'
            f'<text x="{label_w + bw + 8:.1f}" y="{y + 15}" class="val">{t.count}'
            f'<tspan class="muted"> · {pct(t.negative_share)} negative</tspan></text></g>'
        )
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Comments per theme">'
            + "".join(rows) + "</svg>")


def f1_bars(main: Evaluation, main_name: str, codebook: Codebook, baseline: Evaluation | None) -> str:
    """Grouped bars: F1 per code for the main run (and the baseline, if given)."""
    series = [(main_name, main, "s1")] + ([("Keyword baseline", baseline, "s2")] if baseline else [])
    group_h = 14 * len(series) + 16
    label_w, bar_w, w = 170, 400, 640
    h = group_h * len(codebook.codes) + 30
    parts = []
    axis_y = h - 18
    for t in (0, 0.5, 1):  # grid first, so the bars are drawn on top of it
        x = label_w + t * bar_w
        parts.append(f'<line x1="{x}" x2="{x}" y1="0" y2="{axis_y}" class="grid"/>'
                     f'<text x="{x}" y="{axis_y + 14}" text-anchor="middle" class="axis small">{t:g}</text>')
    for gi, code in enumerate(codebook.codes):
        y0 = gi * group_h + 6
        parts.append(f'<text x="{label_w - 12}" y="{y0 + 7 * len(series) + 4}" text-anchor="end" class="axis">'
                     f'{esc(code.label)}</text>')
        for si, (name, ev, cls) in enumerate(series):
            s = next(x for x in ev.per_code if x.code == code.id)
            y = y0 + si * 14
            bw = max(s.f1 * bar_w, 2)
            tip = (f"{code.label} – {name}: F1 {s.f1:.2f}, precision {s.precision:.2f}, "
                   f"recall {s.recall:.2f}, kappa {s.kappa:.2f} ({s.support} human-coded)")
            parts.append(
                f'<g class="hit" data-tip="{esc(tip)}" tabindex="0">'
                f'<rect x="{label_w}" y="{y}" width="{bar_w + 60}" height="14" fill="transparent"/>'
                f'<rect x="{label_w}" y="{y + 1}" width="{bw:.1f}" height="11" rx="3" class="bar {cls}"/>'
                f'<text x="{label_w + bw + 6:.1f}" y="{y + 11}" class="val small">{s.f1:.2f}</text></g>'
            )
    legend = ""
    if baseline:
        legend = (f'<div class="legend"><span><i class="sw s1"></i>{esc(main_name)}</span>'
                  f'<span><i class="sw s2"></i>Keyword baseline</span></div>')
    return legend + (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="F1 score per code">'
                     + "".join(parts) + "</svg>")


# ---------------------------------------------------------------- page

def render(run: RunResult, ev: Evaluation, themes: list[Theme], codebook: Codebook,
           baseline: Evaluation | None = None) -> str:
    total = len(run.comments)
    review = [c for c in run.comments if c.needs_review]
    assigned = sum(len(c.codes) for c in run.comments)
    is_baseline = run.provider == "offline"
    run_name = "Keyword baseline" if is_baseline else f"{run.provider} · {run.model}"

    kpis = [
        (str(total), "comments coded"),
        (str(assigned), "codes assigned"),
        (f"{ev.micro_f1:.2f}", "micro-F1 vs human"),
        (f"{ev.mean_kappa:.2f}", "mean Cohen's kappa"),
        (str(len(review)), "sent to human review"),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for v, k in kpis)

    theme_cards = []
    for t in themes:
        if not t.count:
            continue
        quotes = "".join(f'<li>“{esc(q)}” <span class="muted">{esc(cid)}</span></li>' for cid, q in t.quotes)
        theme_cards.append(f'<article class="card"><h3>{esc(t.label)} <span class="muted">{t.count}</span></h3>'
                           f'<ul class="quotes">{quotes}</ul></article>')
    pairs = co_occurrence(run.comments)
    pairs_html = "".join(f"<li>{esc(codebook.label(a))} + {esc(codebook.label(b))} "
                         f'<span class="muted">{k}×</span></li>' for a, b, k in pairs) or "<li>None</li>"

    score_rows = []
    for s in ev.per_code:
        base = next((b for b in baseline.per_code if b.code == s.code), None) if baseline else None
        base_cell = f"<td>{base.f1:.2f}</td>" if base else ""
        score_rows.append(f"<tr><td>{esc(codebook.label(s.code))}</td><td>{s.support}</td>"
                          f"<td>{s.precision:.2f}</td><td>{s.recall:.2f}</td><td><b>{s.f1:.2f}</b></td>"
                          f"<td>{s.kappa:.2f}</td>{base_cell}</tr>")
    base_head = "<th>Baseline F1</th>" if baseline else ""
    compare_line = ""
    if baseline and not is_baseline:
        compare_line = (f"<p>The keyword baseline reaches micro-F1 <b>{baseline.micro_f1:.2f}</b> and kappa "
                        f"<b>{baseline.mean_kappa:.2f}</b> on the same comments.</p>")

    review_items = []
    for c in review:
        reasons = "".join(f"<li><code>{esc(f.kind)}</code> {esc(f.detail)}</li>" for f in c.flags)
        review_items.append(f'<li><b>{esc(c.id)}</b> {esc(c.text)}<ul class="reasons">{reasons}</ul></li>')
    review_html = "".join(review_items) or "<li>Nothing – every answer passed the checks.</li>"

    disagree_html = "".join(
        f'<li><b>{esc(d["id"])}</b> {esc(d["text"])}<br><span class="muted">'
        f'human: {esc(", ".join(d["human"]) or "none")} · model: {esc(", ".join(d["model"]) or "none")}</span></li>'
        for d in ev.disagreements
    ) or "<li>None.</li>"

    rows = []
    for c in run.comments:
        chips = "".join(f'<span class="chip" title="confidence {vc.confidence:.2f}">'
                        f"{esc(codebook.label(vc.code))}</span>" for vc in c.codes)
        flag = '<span class="flag">review</span>' if c.needs_review else ""
        rows.append(f'<tr data-codes="{esc(" ".join(sorted(c.code_ids)))}" data-review="{int(c.needs_review)}">'
                    f"<td>{esc(c.id)}</td><td>{highlight(c.text, [vc.evidence for vc in c.codes])}</td>"
                    f"<td>{chips}{flag}</td><td>{esc(c.sentiment)}</td></tr>")
    filters = '<button class="f on" data-f="all">All</button>' + "".join(
        f'<button class="f" data-f="{esc(c.id)}">{esc(c.label)}</button>' for c in codebook.codes
    ) + '<button class="f" data-f="review">Needs review</button>'

    stats = run.stats
    meta = (f"{esc(run_name)} · {esc(run.created)} · {stats.batches} batches, {stats.api_calls} API calls, "
            f"{stats.cache_hits} from cache, {stats.retries} retries")
    data_json = json.dumps({"metrics": ev.to_dict()}, ensure_ascii=False).replace("</", "<\\/")

    return TEMPLATE.format(
        run_name=esc(run_name), meta=meta, kpis=kpi_html,
        theme_chart=theme_bars(themes, total), theme_cards="".join(theme_cards), pairs=pairs_html,
        f1_chart=f1_bars(ev, run_name, codebook, baseline if not is_baseline else None),
        micro=f"{ev.micro_f1:.2f}", macro=f"{ev.macro_f1:.2f}", kappa=f"{ev.mean_kappa:.2f}",
        exact=pct(ev.exact_match), senti=pct(ev.sentiment_accuracy), compare=compare_line,
        score_rows="".join(score_rows), base_head=base_head,
        review_count=len(review), review=review_html, disagreements=disagree_html,
        filters=filters, rows="".join(rows), data_json=data_json,
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feedback Coder report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #fcfcfb; --card: #ffffff; --line: #e4e3df; --text: #0b0b0b; --muted: #52514e;
  --s1: #2a78d6; --s2: #eb6834; --mark: #fff1b8; --flag: #d03b3b; --chip: #eef3fb;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --bg: #1a1a19; --card: #232321; --line: #383835; --text: #f0efec; --muted: #c3c2b7;
    --s1: #3987e5; --s2: #d95926; --mark: #5a4a12; --flag: #e66767; --chip: #26324a;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #1a1a19; --card: #232321; --line: #383835; --text: #f0efec; --muted: #c3c2b7;
  --s1: #3987e5; --s2: #d95926; --mark: #5a4a12; --flag: #e66767; --chip: #26324a;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 1.9rem; margin: 0 0 4px; }}
h2 {{ font-size: 1.3rem; margin: 48px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 1rem; margin: 0 0 8px; }}
.muted {{ color: var(--muted); font-weight: normal; }}
.lead {{ color: var(--muted); max-width: 70ch; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0; }}
.kpi {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
.kpi b {{ display: block; font-size: 1.7rem; line-height: 1.2; }}
.kpi span {{ color: var(--muted); font-size: .9rem; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; margin-top: 16px; }}
.quotes {{ margin: 0; padding-left: 18px; font-size: .92rem; }}
.quotes li {{ margin-bottom: 4px; }}
svg {{ width: 100%; height: auto; display: block; }}
.chart {{ overflow-x: auto; }} .chart svg {{ min-width: 560px; }}
@media (max-width: 640px) {{
  #comments thead {{ display: none; }}
  #comments tr {{ display: block; padding: 10px 0; border-bottom: 1px solid var(--line); }}
  #comments td {{ display: block; border: 0; padding: 2px 0; }}
  #comments td:first-child {{ color: var(--muted); font-size: .85rem; }}
}}
svg text {{ fill: var(--text); font-size: 13px; }}
svg .axis {{ fill: var(--muted); }}
svg .small {{ font-size: 11px; }}
svg .muted {{ fill: var(--muted); }}
svg .grid {{ stroke: var(--line); stroke-width: 1; }}
.bar.s1, .sw.s1 {{ fill: var(--s1); background: var(--s1); }}
.bar.s2, .sw.s2 {{ fill: var(--s2); background: var(--s2); }}
.hit:hover .bar, .hit:focus .bar {{ opacity: .8; }}
.hit:focus {{ outline: none; }}
.legend {{ display: flex; gap: 16px; font-size: .9rem; color: var(--muted); margin-bottom: 6px; }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
#tip {{ position: fixed; pointer-events: none; background: var(--text); color: var(--bg); padding: 6px 10px;
  border-radius: 8px; font-size: .85rem; max-width: 320px; opacity: 0; transition: opacity .1s; z-index: 10; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .93rem; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; }}
mark {{ background: var(--mark); color: inherit; border-radius: 3px; padding: 0 2px; }}
.chip {{ display: inline-block; background: var(--chip); border-radius: 999px; padding: 1px 9px;
  margin: 0 4px 4px 0; font-size: .82rem; white-space: nowrap; }}
.flag {{ display: inline-block; border: 1px solid var(--flag); color: var(--flag); border-radius: 999px;
  padding: 0 8px; font-size: .78rem; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }}
.f {{ border: 1px solid var(--line); background: var(--card); color: var(--text); border-radius: 999px;
  padding: 4px 12px; font: inherit; font-size: .85rem; cursor: pointer; }}
.f.on {{ background: var(--text); color: var(--bg); border-color: var(--text); }}
ul.list {{ padding-left: 18px; }} ul.list > li {{ margin-bottom: 10px; }}
.reasons {{ font-size: .88rem; color: var(--muted); }}
code {{ font-size: .85em; }}
.steps {{ counter-reset: s; list-style: none; padding: 0; display: grid; gap: 8px; }}
.steps li {{ counter-increment: s; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px 10px 44px; position: relative; }}
.steps li::before {{ content: counter(s); position: absolute; left: 14px; top: 10px; font-weight: 700; color: var(--s1); }}
.theme-btn {{ float: right; }}
footer {{ margin-top: 48px; color: var(--muted); font-size: .85rem; }}
</style>
</head>
<body>
<div id="tip" role="tooltip"></div>
<main>
<button class="f theme-btn" id="theme" aria-label="Switch between light and dark theme">Light / dark</button>
<h1>Feedback Coder</h1>
<p class="lead">LLM-assisted qualitative coding of user feedback for a fictional budgeting app – with every
answer checked before it is trusted. <br><span class="muted">{meta}</span></p>
<div class="kpis">{kpis}</div>

<h2>What users talk about</h2>
<p class="lead">Comments per theme. One comment can belong to several themes. Hover a bar for details.</p>
<div class="chart">{theme_chart}</div>
<div class="cards">{theme_cards}</div>
<div class="card" style="margin-top:12px"><h3>Themes that appear together</h3><ul class="quotes">{pairs}</ul></div>

<h2>Does it agree with a human coder?</h2>
<p class="lead">Every comment was also coded by hand with the same codebook. Micro-F1 <b>{micro}</b> ·
macro-F1 <b>{macro}</b> · mean kappa <b>{kappa}</b> · identical code set for <b>{exact}</b> of comments ·
sentiment correct for <b>{senti}</b>.</p>
{compare}
<div class="chart">{f1_chart}</div>
<div class="table-wrap"><table>
<thead><tr><th>Code</th><th>Human-coded</th><th>Precision</th><th>Recall</th><th>F1</th><th>Kappa</th>{base_head}</tr></thead>
<tbody>{score_rows}</tbody></table></div>

<h2>Sent back to a human ({review_count})</h2>
<p class="lead">Answers that failed a check: a code outside the codebook, a quote that is not in the comment,
low confidence, or a missing answer. These are never used silently.</p>
<ul class="list">{review}</ul>

<h2>Where model and human disagree</h2>
<p class="lead">The most useful list for improving the codebook: each disagreement is either a model error or an
unclear rule.</p>
<ul class="list">{disagreements}</ul>

<h2>All comments</h2>
<p class="lead">Highlighted text is the evidence quoted for each code.</p>
<div class="filters" role="group" aria-label="Filter comments">{filters}</div>
<div class="table-wrap"><table id="comments">
<thead><tr><th>ID</th><th>Comment</th><th>Codes</th><th>Sentiment</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>How it works</h2>
<ol class="steps">
<li><b>Codebook.</b> Eight codes, each with a definition, include/exclude rules and examples (codebook.yaml).</li>
<li><b>Prompt.</b> The codebook and a batch of 10 comments go to the model, which must answer in a fixed JSON shape.</li>
<li><b>Validate.</b> The answer is checked against a schema; if it is invalid, the error is sent back and the model retries.</li>
<li><b>Verify.</b> Unknown codes and quotes that do not appear in the comment are dropped; low confidence goes to review.</li>
<li><b>Evaluate.</b> The result is compared with human labels: precision, recall, F1 and Cohen's kappa per code.</li>
<li><b>Report.</b> Themes, quotes, disagreements and the review queue, in this page.</li>
</ol>
<footer>Sample data: 60 fictional comments written for this demo. Generated by feedback-coder.</footer>
</main>
<script type="application/json" id="data">{data_json}</script>
<script>
(() => {{
  const tip = document.getElementById('tip');
  const show = (el, x, y) => {{ tip.textContent = el.dataset.tip; tip.style.opacity = 1;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(x + 12, innerWidth - r.width - 8) + 'px'; tip.style.top = (y + 14) + 'px'; }};
  document.querySelectorAll('.hit').forEach(el => {{
    el.addEventListener('mousemove', e => show(el, e.clientX, e.clientY));
    el.addEventListener('mouseleave', () => tip.style.opacity = 0);
    el.addEventListener('focus', () => {{ const b = el.getBoundingClientRect(); show(el, b.left, b.top); }});
    el.addEventListener('blur', () => tip.style.opacity = 0);
  }});
  const rows = [...document.querySelectorAll('#comments tbody tr')];
  document.querySelectorAll('.filters .f').forEach(btn => btn.addEventListener('click', () => {{
    document.querySelectorAll('.filters .f').forEach(b => b.classList.toggle('on', b === btn));
    const f = btn.dataset.f;
    rows.forEach(r => r.hidden = !(f === 'all' || (f === 'review' ? r.dataset.review === '1'
      : r.dataset.codes.split(' ').includes(f))));
  }}));
  const root = document.documentElement;
  try {{ const t = localStorage.getItem('fc-theme'); if (t) root.dataset.theme = t; }} catch (e) {{}}
  document.getElementById('theme').addEventListener('click', () => {{
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try {{ localStorage.setItem('fc-theme', root.dataset.theme); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""
