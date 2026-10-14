"""Style and script of the explorer page.

All text from the data reaches the DOM through ``textContent``. Strings given
to Plotly pass through ``plain`` first, because Plotly reads a small HTML
subset in labels and a Wikipedia title could contain it.
"""

# ruff: noqa: E501  (embedded CSS and JavaScript, wrapped for the browser not the linter)

from __future__ import annotations

from typing import Final

__all__ = ["EXPLORER_CSS", "EXPLORER_JS"]

EXPLORER_CSS: Final[str] = """
:root { --bg:#fbfaf7; --fg:#1d2a33; --muted:#5d6b75; --line:#d9d6cc; --accent:#2a6f97;
  --warn-bg:#fff1d6; --warn-fg:#6b3d00; --flag:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161b20; --fg:#e6e9eb; --muted:#9aa7b0; --line:#2c353d; --accent:#6db3d9;
    --warn-bg:#3d2c0a; --warn-fg:#ffd699; --flag:#ff7b6b; }
}
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
header, main { max-width:1100px; margin:0 auto; padding:0 16px; }
h1 { margin:24px 0 4px; } h2 { margin:28px 0 8px; font-size:1.2em; }
.meta, #sens-note { color:var(--muted); }
.banner { background:var(--warn-bg); color:var(--warn-fg); padding:10px 14px; border-radius:6px; }
.scroll { overflow-x:auto; max-height:420px; overflow-y:auto; border:1px solid var(--line); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); }
th { position:sticky; top:0; background:var(--bg); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.pick { cursor:pointer; } tr.pick:hover { background:rgba(42,111,151,.12); }
input[type=search] { width:100%; max-width:360px; padding:8px; margin-bottom:8px; box-sizing:border-box; }
.chart { width:100%; } #chart-sens { min-height:420px; } .flag { color:var(--flag); font-weight:600; }
"""

EXPLORER_JS: Final[str] = r"""
(function () {
  var data = JSON.parse(document.getElementById("report-data").textContent);
  var gens = data.generals;

  function cell(tr, text, num) {
    var td = document.createElement("td");
    td.textContent = text === null || text === undefined ? "n/a" : String(text);
    if (num) { td.className = "num"; }
    tr.appendChild(td);
    return td;
  }
  function fmt(x, d) { return x === null || x === undefined ? "n/a" : Number(x).toFixed(d); }
  function plain(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function overlapText(v) { return v === true ? "yes" : (v === false ? "no" : "n/a"); }
  function rankText(g) { return g.rank === null || g.rank === undefined ? "n/a" : String(g.rank); }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function themed() {
    var line = css("--line");
    return {
      font: {color: css("--fg")},
      grid: {gridcolor: line, zerolinecolor: line, linecolor: line},
      accent: css("--accent"), flag: css("--flag"), muted: css("--muted")
    };
  }
  function ci(g) { return fmt(g.skill_mean, 2) + " [" + fmt(g.skill_ci95_lo, 2) + ", " + fmt(g.skill_ci95_hi, 2) + "]"; }

  function renderTable(query) {
    var body = document.querySelector("#general-table tbody");
    body.textContent = "";
    var q = query.trim().toLowerCase();
    var shown = 0;
    gens.forEach(function (g) {
      if (q && g.name.toLowerCase().indexOf(q) < 0) { return; }
      var tr = document.createElement("tr");
      tr.className = "pick";
      tr.tabIndex = 0;
      cell(tr, g.headline ? rankText(g) : "unranked", true);
      cell(tr, g.name, false);
      cell(tr, g.n_battles, true);
      cell(tr, fmt(g.total_war, 2), true);
      cell(tr, fmt(g.war_sd, 2), true);
      cell(tr, ci(g), false);
      var flag = cell(tr, overlapText(g.overlaps_next_rank), false);
      if (g.overlaps_next_rank === true) { flag.className = "flag"; }
      tr.addEventListener("click", function () { showDetail(g); });
      tr.addEventListener("keydown", function (ev) { if (ev.key === "Enter") { showDetail(g); } });
      body.appendChild(tr);
      shown += 1;
    });
    document.getElementById("no-match").hidden = shown > 0;
  }

  function showDetail(g) {
    document.getElementById("detail").hidden = false;
    document.getElementById("detail-name").textContent = g.name;
    var lead = g.headline ? "Rank " + rankText(g) + ". " :
      "Under-determined: fewer than " + data.min_battles + " credited battles, so no rank is shown. ";
    document.getElementById("detail-summary").textContent = lead +
      g.n_battles + " credited battles (effective battles " + fmt(g.effective_battles, 1) +
      ", weight-summed), expected total WAR " + fmt(g.total_war, 2) +
      " (sd " + fmt(g.war_sd, 2) + "), skill " + ci(g) + ".";
    var body = document.querySelector("#battle-table tbody");
    body.textContent = "";
    g.battles.forEach(function (b) {
      var tr = document.createElement("tr");
      cell(tr, b.battle, false); cell(tr, b.year, true); cell(tr, b.polity, false);
      cell(tr, b.side_outcome, false); cell(tr, b.command_role, false);
      cell(tr, fmt(b.attribution_weight, 2), true); cell(tr, fmt(b.expected_war, 3), true);
      cell(tr, fmt(b.expected_score_model, 3), true);
      cell(tr, fmt(b.expected_score_replacement, 3), true);
      cell(tr, fmt(b.actual_outcome, 1), true);
      body.appendChild(tr);
    });
    document.getElementById("detail").scrollIntoView({behavior: "smooth"});
  }

  function warChart() {
    var host = document.getElementById("chart-war");
    var top = gens.filter(function (g) { return g.headline && g.war_normal_approx_lo !== null; }).slice(0, 50);
    if (typeof Plotly === "undefined" || top.length === 0) {
      host.textContent = "No intervals to draw.";
      return;
    }
    var t = themed();
    function trace(match, name, color) {
      var rows = top.filter(function (g) { return match(g.overlaps_next_rank); });
      return {
        type: "scatter", mode: "markers", name: name,
        x: rows.map(function (g) { return g.total_war; }),
        y: rows.map(function (g) { return rankText(g) + ". " + plain(g.name); }),
        error_x: {type: "data", visible: true, color: color,
          array: rows.map(function (g) { return g.war_normal_approx_hi - g.total_war; }),
          arrayminus: rows.map(function (g) { return g.total_war - g.war_normal_approx_lo; })},
        marker: {color: color}
      };
    }
    var traces = [
      trace(function (v) { return v === true; }, "overlaps next listed general", t.flag),
      trace(function (v) { return v === false; }, "separate from next listed general", t.accent),
      trace(function (v) { return v === null || v === undefined; }, "n/a (no next listed general or no interval)", t.muted)
    ].filter(function (tr) { return tr.x.length > 0; });
    var height = 20 * top.length + 120;
    host.style.height = height + "px";
    var axis = {gridcolor: t.grid.gridcolor, zerolinecolor: t.grid.zerolinecolor, linecolor: t.grid.linecolor};
    Plotly.newPlot(host, traces, {
      xaxis: Object.assign({title: {text: "Expected total WAR (bars: normal approximation)"}}, axis),
      yaxis: Object.assign({autorange: "reversed", type: "category"}, axis),
      font: {color: t.font.color},
      margin: {l: 190, r: 20, t: 20, b: 50}, height: height,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
    }, {responsive: true});
  }

  function sensChart(variant) {
    var host = document.getElementById("chart-sens");
    var table = data.sensitivity.ranks[variant] || {};
    var rows = gens.filter(function (g) { return g.headline && g.rank !== null && table[String(g.general_id)] !== undefined; });
    if (typeof Plotly === "undefined" || rows.length === 0) {
      host.textContent = "No ranks to compare.";
      return;
    }
    var t = themed();
    var axis = {gridcolor: t.grid.gridcolor, zerolinecolor: t.grid.zerolinecolor, linecolor: t.grid.linecolor};
    var hi = Math.max.apply(null, rows.map(function (g) { return Math.max(g.rank, table[String(g.general_id)]); }));
    Plotly.newPlot(host, [
      {type: "scatter", mode: "lines", x: [0, hi], y: [0, hi], hoverinfo: "skip",
        line: {color: t.muted, dash: "dash"}, showlegend: false},
      {type: "scatter", mode: "markers", showlegend: false,
        x: rows.map(function (g) { return g.rank; }),
        y: rows.map(function (g) { return table[String(g.general_id)]; }),
        text: rows.map(function (g) { return plain(g.name); }), marker: {color: t.accent}}
    ], {
      xaxis: Object.assign({title: {text: "Published rank"}}, axis),
      yaxis: Object.assign({title: {text: "Rank under " + plain(variant)}, rangemode: "tozero"}, axis),
      font: {color: t.font.color},
      margin: {t: 20}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
    }, {responsive: true});
  }

  function redraw() {
    warChart();
    var select = document.getElementById("variant");
    if (data.sensitivity && select.value) { sensChart(select.value); }
  }

  function sensitivity() {
    var note = document.getElementById("sens-note");
    var select = document.getElementById("variant");
    if (!data.sensitivity) {
      note.textContent = "Not available: " + (data.evaluation_unavailable || "no evaluation report was read") + ".";
      select.hidden = true;
      select.parentNode.hidden = true;
      return;
    }
    var info = {};
    data.sensitivity.variants.forEach(function (v) { info[v.name] = v; });
    note.textContent = "Ranks come from the evaluate stage's refits, which rank by their own scores. " +
      "Points on the dashed line keep their published rank.";
    var names = Object.keys(data.sensitivity.ranks);
    names.forEach(function (n) {
      var opt = document.createElement("option");
      var v = info[n];
      opt.value = n;
      opt.textContent = n + (v && v.spearman_rho !== null ? " (rho " + fmt(v.spearman_rho, 3) + ")" : "");
      select.appendChild(opt);
    });
    select.addEventListener("change", function () { sensChart(select.value); });
    if (names.length > 0) { sensChart(names[0]); }
  }

  document.getElementById("search").addEventListener("input", function (ev) { renderTable(ev.target.value); });
  renderTable("");
  warChart();
  sensitivity();
  if (window.matchMedia) {
    var scheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (scheme.addEventListener) { scheme.addEventListener("change", redraw); }
  }
})();
"""
