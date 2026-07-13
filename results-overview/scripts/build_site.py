#!/usr/bin/env python3
"""Static-site generator for the VLM Council evaluation Results repo.

Scans every approach folder under the repo root and builds a `site/`
directory containing:
  - index.html            landing page with one card per approach
  - <approach>/            per-approach copy of the consolidated evaluation
                           report (HTML) and its plots.

Each approach has ONE consolidated evaluation report per approach
(evaluation/report.html + evaluation/plots + *_metrics.json). It combines
ground-truth statistics, approach dynamics, and the LLM-as-Judge verdicts
in a single document.

The Initial Approach is a special case: it ships per-variant CSVs under
llm_council_evals/ instead of a single evaluation report.

No third-party dependencies - standard library only.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE = REPO_ROOT / "site"

# Shared theme tokens (mirrors the existing evaluation report.html look).
THEME_CSS = """
:root {
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
  --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
  --success: #22c55e; --error: #ef4444; --warning: #f59e0b;
  --card-bg: #1e2130;
}
[data-theme="light"] {
  --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0;
  --text: #1e293b; --muted: #64748b; --accent: #4f46e5;
  --success: #16a34a; --error: #dc2626; --warning: #d97706;
  --card-bg: #f1f5f9;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif; font-size: 15px;
  line-height: 1.6; padding: 2.5rem 1.25rem; }
.wrap { max-width: 1100px; margin: 0 auto; }
header.page { margin-bottom: 2rem; }
h1 { font-size: 1.9rem; margin-bottom: .4rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 1rem; color: var(--accent); }
p.lead { color: var(--muted); max-width: 70ch; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.25rem; }
.card { background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.25rem 1.4rem; }
.card h3 { font-size: 1.1rem; margin-bottom: .3rem; }
.card .headline { color: var(--muted); font-size: .9rem; margin-bottom: .9rem; }
.links a { display: inline-flex; align-items: center; gap: .4rem;
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); text-decoration: none; padding: .4rem .8rem;
  border-radius: 8px; font-size: .85rem; margin: .2rem .3rem .2rem 0; }
.links a:hover { border-color: var(--accent); color: var(--accent); }
.links a.disabled { opacity: .4; pointer-events: none; }
.badge { display: inline-block; background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px;
  padding: .1rem .5rem; font-size: .75rem; color: var(--muted); }
.toggle { position: fixed; top: 1rem; right: 1rem; cursor: pointer;
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); border-radius: 8px; padding: .4rem .7rem; font-size: .85rem; }
pre { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.25rem; overflow-x: auto;
  font-size: 13px; line-height: 1.5; white-space: pre; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
  border-top: 1px solid var(--border); padding-top: 1rem; }
img.plot { max-width: 100%; border: 1px solid var(--border);
  border-radius: 10px; margin: .8rem 0; background: #fff; }
a.back { color: var(--accent); text-decoration: none; font-size: .9rem; }
"""

THEME_TOGGLE_JS = """
<script>
(function(){
  var KEY='vlm-theme';
  var saved=localStorage.getItem(KEY);
  if(saved){document.documentElement.setAttribute('data-theme',saved);}
  function t(){
    var cur=document.documentElement.getAttribute('data-theme')==='light'?'':'light';
    if(cur){document.documentElement.setAttribute('data-theme',cur);}
    else{document.documentElement.removeAttribute('data-theme');}
    localStorage.setItem(KEY,cur);
  }
  window.addEventListener('DOMContentLoaded',function(){
    var b=document.getElementById('themeToggle');
    if(b){b.addEventListener('click',t);}
  });
})();
</script>
"""


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"UTF-8\"/>\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{THEME_CSS}</style>\n</head>\n<body>\n"
        "<button class=\"toggle\" id=\"themeToggle\">Theme</button>\n"
        f"<div class=\"wrap\">\n{body}\n</div>\n{THEME_TOGGLE_JS}\n</body>\n</html>\n"
    )


_ASCII_MAP = {
    "\u2192": "->", "\u27f6": "->", "\u2190": "<-",
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2022": "*",
}

# HTML-entity forms of the same fancy punctuation (reports may encode them).
# Keys are built from an escaped ampersand so this table stays ASCII-clean.
_AMP = "&"  # &
_ENTITY_MAP = {
    _AMP + "mdash;": "-", _AMP + "ndash;": "-", _AMP + "minus;": "-",
    _AMP + "rarr;": "->", _AMP + "larr;": "<-", _AMP + "bull;": "*",
}


def _ascii(text: str) -> str:
    """Fancy punctuation -> ASCII so the published site never renders em/en
    dashes, arrows, or bullets even if an upstream report contains them
    (as literal characters or as HTML entities)."""
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    for k, v in _ENTITY_MAP.items():
        text = text.replace(k, v)
    return text


def _find_judge_report(approach_dir: Path) -> Path | None:
    """The judge HTML report may be report.html or a variant-named .html."""
    evald = approach_dir / "evaluation"
    if not evald.is_dir():
        return None
    cand = evald / "report.html"
    if cand.exists():
        return cand
    htmls = sorted(evald.glob("*.html"))
    return htmls[0] if htmls else None


def build() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    # Approaches excluded from the navigation page. The Initial Approach is a
    # two-stage Ollama system whose results are per-variant CSVs rather than a
    # judge report + GT analysis, so it is not surfaced here.
    EXCLUDED = {"VLM Initial Approach"}

    approaches = sorted(
        p for p in REPO_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("VLM ") and p.name not in EXCLUDED
    )

    cards = []
    for ap in approaches:
        name = ap.name
        slug = name.replace(" ", "_").replace("+", "plus")
        out_dir = SITE / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        links = []

        # --- Consolidated evaluation report (GT stats + dynamics + LLM-as-Judge) ---
        judge = _find_judge_report(ap)
        if judge:
            rp = out_dir / "report.html"
            report_html = _ascii(judge.read_text(errors="ignore"))
            plots_src = ap / "evaluation" / "plots"
            if plots_src.is_dir():
                shutil.copytree(plots_src, out_dir / "plots", dirs_exist_ok=True)
            rp.write_text(report_html)
            links.append(f'<a href="{slug}/report.html">Evaluation report</a>')

        # --- Initial Approach special case: per-variant CSVs ---
        csv_dir = ap / "llm_council_evals"
        if csv_dir.is_dir():
            shutil.copytree(csv_dir, out_dir / "llm_council_evals", dirs_exist_ok=True)
            links.append(f'<a href="{slug}/llm_council_evals/">Per-variant CSVs</a>')

        card = (
            f'<div class="card"><h3>{html.escape(name)}</h3>'
            f'<div class="links">{"".join(links)}</div></div>'
        )
        cards.append(card)

    body = (
        '<header class="page"><h1>VLM Council - GeoRC Evaluation</h1>'
        '<p class="lead">Country-geolocation evaluation across all VLM Council '
        'approaches on the GeoRC benchmark. Each approach has one consolidated '
        'evaluation report combining ground-truth statistics, approach dynamics, '
        'and an LLM-as-Judge (Qwen 3.6) qualitative evaluation.</p></header>'
        '<h2>Approaches</h2>'
        f'<div class="grid">{"".join(cards)}</div>'
        '<footer>Generated from the evaluation Results by '
        '<code>scripts/build_site.py</code>. One consolidated evaluation report '
        'per approach.</footer>'
    )
    (SITE / "index.html").write_text(_page("VLM Council - GeoRC Evaluation", body))
    # .nojekyll so GitHub Pages serves _underscore dirs and raw assets verbatim
    (SITE / ".nojekyll").write_text("")
    print(f"Built site/ with {len(approaches)} approaches -> {SITE}")


if __name__ == "__main__":
    build()
