"""The self-contained ``explorer.html``: data inlined as escaped JSON, plotly.js inlined once."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from string import Template
from typing import Any

from pipeline.reporters.explorer_script import EXPLORER_CSS, EXPLORER_JS

__all__ = ["embed_json", "load_plotly_js", "render_explorer"]

# The page and the script contain no dollar sign other than these placeholders,
# so string.Template needs no escaping and cannot rescan the inlined plotly.js.
_PAGE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>General WAR explorer</title>
<style>${css}</style>
</head>
<body>
<header>
<h1>General WAR explorer</h1>
<p class="meta">Model run ${run_id}. Expected WAR is wins above replacement under the fitted \
model. Expected score is P(win) plus half of P(indecisive), not P(win).</p>
${banner}
</header>
<main>
<section id="ranking">
<h2>Search generals</h2>
${cap_note}
<input id="search" type="search" placeholder="Search by name" aria-label="Search generals">
<div class="scroll"><table id="general-table"><thead><tr>
<th>Rank</th><th>General</th><th>Credited battles</th><th>Expected WAR</th><th>WAR sd</th>
<th>WAR (95% interval)</th><th>Skill (95% CI)</th><th>P(above next ranked)</th>
<th>Overlaps next listed general</th></tr></thead>
<tbody></tbody></table></div>
<p id="no-match" hidden>No general matches.</p>
</section>
<section id="detail" hidden>
<h2 id="detail-name"></h2>
<p id="detail-summary"></p>
<div class="scroll"><table id="battle-table"><thead><tr>
<th>Battle</th><th>Year</th><th>Side</th><th>Outcome</th><th>Role</th><th>Weight</th>
<th>Expected WAR</th><th>Expected score (model)</th><th>Expected score (replacement)</th>
<th>Actual outcome</th></tr></thead><tbody></tbody></table></div>
</section>
<section id="uncertainty">
<h2>Where the order is fragile</h2>
<p>${overlap_note}</p>
<div id="chart-war" class="chart"></div>
</section>
<section id="sensitivity">
<h2>Sensitivity comparison</h2>
<p id="sens-note"></p>
<label for="variant">Variant <select id="variant"></select></label>
<div id="chart-sens" class="chart"></div>
</section>
<section id="disclosures">
<h2>Disclosures</h2>
<ul>
${disclosures}
</ul>
</section>
</main>
<script type="application/json" id="report-data">${data}</script>
<script>${plotly}</script>
<script>${script}</script>
</body>
</html>
"""
)


def embed_json(payload: Any) -> str:
    """Serialise JSON so it cannot close or open an HTML element when inlined.

    ``<``, ``>`` and ``&`` become unicode escapes, which JSON.parse reads back
    as the same characters; non-ASCII is escaped by ``ensure_ascii``.

    Args:
        payload: A JSON-serialisable value.

    Returns:
        Text safe inside ``<script type="application/json">``.
    """
    raw = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def load_plotly_js() -> str:
    """The bundled plotly.js source, guarded against closing its own script tag.

    Returns:
        The JavaScript text.
    """
    from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]

    return str(get_plotlyjs()).replace("</script", "<\\/script")


def _items(lines: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"<li>{html.escape(line, quote=False)}</li>" for line in lines)


def _cap_note(payload: Mapping[str, Any]) -> str:
    shown, total = payload.get("n_generals_shown"), payload.get("n_generals_total")
    if not isinstance(shown, int) or not isinstance(total, int) or shown >= total:
        return ""
    return (
        f'<p class="meta" id="cap-note">Showing {shown} of {total} generals, headline ranks '
        "first. rankings.csv and rankings.json hold all of them.</p>"
    )


def render_explorer(payload: Mapping[str, Any], *, plotly_js: str | None = None) -> str:
    """Render the explorer page.

    Disclosures and the provisional banner are written as static HTML, escaped,
    so they show without scripts; everything else is built from the inlined
    JSON with ``textContent``.

    Args:
        payload: Output of ``exports.build_payload``.
        plotly_js: The plotly.js source to inline; fetched when None.

    Returns:
        The HTML document.
    """
    reasons = payload.get("provisional_reasons") or []
    banner = (
        '<div class="banner" role="alert"><strong>PROVISIONAL. Do not publish this ranking.'
        f"</strong><ul>{_items(reasons)}</ul></div>"
        if payload.get("provisional")
        else ""
    )
    return _PAGE.substitute(
        css=EXPLORER_CSS,
        run_id=html.escape(str(payload.get("model_run_id")), quote=False),
        banner=banner,
        cap_note=_cap_note(payload),
        overlap_note=html.escape(str(payload.get("overlap_method", "")), quote=False),
        disclosures=_items(list(payload.get("disclosures") or [])),
        data=embed_json(payload),
        plotly=plotly_js if plotly_js is not None else load_plotly_js(),
        script=EXPLORER_JS,
    )
