"""The explorer's "Data coverage and caveats" tab: markup and plotly script.

The six panels are drawn from ``data.coverage``, the same dictionary that feeds
the report's PNGs. Everything text-like from the data goes through ``plain`` or
``textContent``. The panels draw when the tab is first shown, because Plotly
cannot size a chart inside a hidden element.
"""

# ruff: noqa: E501  (embedded HTML and JavaScript, wrapped for the browser not the linter)

from __future__ import annotations

from typing import Final

__all__ = ["COVERAGE_JS", "COVERAGE_TAB_ID", "COVERAGE_TAB_HTML", "COVERAGE_TITLE"]

COVERAGE_TAB_ID: Final[str] = "tab-coverage"
COVERAGE_TITLE: Final[str] = "Data coverage and caveats"

COVERAGE_TAB_HTML: Final[str] = f"""<div id="{COVERAGE_TAB_ID}" role="tabpanel" hidden>
<h2>{COVERAGE_TITLE}</h2>
<p class="meta">Where the corpus is thin, and what a rank does and does not rest on. Each panel is drawn from the same queries as the coverage sentences under Disclosures.</p>
<h3>Corpus timeline</h3><div id="chart-cov-timeline" class="chart"></div>
<h3>Comparison components</h3><div id="chart-cov-components" class="chart"></div>
<h3>Credit source</h3><div id="chart-cov-credit_source" class="chart"></div>
<h3>Information against rank</h3><div id="chart-cov-info_vs_rank" class="chart"></div>
<h3>Troop provenance</h3><div id="chart-cov-troop_provenance" class="chart"></div>
<div id="chart-cov-troop_scatter" class="chart"></div>
<h3>Fame bias</h3><div id="chart-cov-fame_bias" class="chart"></div>
</div>"""

COVERAGE_JS: Final[str] = r"""
(function () {
  var data = JSON.parse(document.getElementById("report-data").textContent);
  var cov = data.coverage || {};
  var ERA = {"ancient": "#0072B2", "medieval": "#E69F00", "early modern": "#009E73",
    "Revolutionary and Napoleonic": "#D55E00", "later nineteenth century": "#CC79A7",
    "twentieth century and later": "#56B4E9"};
  var METHOD = {llm: "#0072B2", rule_single: "#D55E00", default_split: "#E69F00", manual: "#009E73", other: "#999999"};
  var drawn = false;

  function plain(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function layout(extra) {
    var line = css("--line");
    var axis = {gridcolor: line, zerolinecolor: line, linecolor: line};
    var base = {font: {color: css("--fg")}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      margin: {l: 60, r: 60, t: 20, b: 50}, xaxis: axis, yaxis: axis};
    Object.keys(extra).forEach(function (k) {
      base[k] = (typeof extra[k] === "object" && !Array.isArray(extra[k]) && base[k]) ? Object.assign({}, base[k], extra[k]) : extra[k];
    });
    return base;
  }
  function host(id) { return document.getElementById("chart-cov-" + id); }
  function missing(id) { host(id).textContent = "Not available: the data for this panel was missing when the report was built."; }
  function plot(id, traces, extra) { Plotly.newPlot(host(id), traces, layout(extra), {responsive: true}); }

  function timeline(p) {
    var eras = Object.keys(ERA);
    var traces = eras.map(function (e) {
      var rows = p.bins.filter(function (b) { return b.era === e; });
      return {type: "bar", name: e, x: rows.map(function (b) { return b.start + 12.5; }),
        y: rows.map(function (b) { return b.n; }), width: 25, marker: {color: ERA[e]}};
    }).filter(function (t) { return t.x.length > 0; });
    var shapes = p.empty_spans.map(function (s) {
      return {type: "rect", xref: "x", yref: "paper", x0: s[0], x1: s[1] + 1, y0: 0, y1: 1,
        fillcolor: "#999999", opacity: 0.25, line: {width: 0}, layer: "below"};
    });
    var notes = p.empty_spans.filter(function (s) { return s[1] - s[0] >= 100; }).map(function (s) {
      return {x: (s[0] + s[1]) / 2, y: 0.95, xref: "x", yref: "paper", text: "no battles", showarrow: false};
    });
    plot("timeline", traces, {barmode: "stack", shapes: shapes, annotations: notes,
      legend: {title: {text: "Coloured by the era of each bin's midpoint"}},
      xaxis: {title: {text: "Year"}}, yaxis: {title: {text: "Credited battles per 25 years"}}});
  }

  function components(p) {
    var idx = p.sizes.map(function (_, i) { return i; });
    var bars = {type: "bar", name: "generals", x: idx, y: p.sizes,
      marker: {color: idx.map(function (i) { return i === 0 ? "#D55E00" : "#9db4c0"; })}};
    var marks = {type: "scatter", mode: "markers", name: "top-ranked generals",
      x: p.markers.map(function (m) { return m.component; }),
      y: p.markers.map(function (m) { return p.sizes[m.component]; }),
      text: p.markers.map(function (m) { return m.rank + ". " + plain(m.name); }),
      marker: {symbol: "triangle-up", size: 10, color: css("--fg")}};
    plot("components", [bars, marks], {xaxis: {title: {text: "Component (0 is the largest)"}, type: "category"},
      yaxis: {title: {text: "Generals"}, type: "log"}});
  }

  function creditSource(p) {
    var labels = p.rows.map(function (r) { return r.rank + ". " + plain(r.name); });
    var traces = p.methods.map(function (m) {
      return {type: "bar", orientation: "h", name: m, y: labels,
        x: p.rows.map(function (r) { return r.weights[m] || 0; }), marker: {color: METHOD[m]}};
    });
    var height = 22 * p.rows.length + 120;
    host("credit_source").style.height = height + "px";
    plot("credit_source", traces, {barmode: "stack", height: height, margin: {l: 190},
      xaxis: {title: {text: "Credited weight"}}, yaxis: {autorange: "reversed", type: "category"}});
  }

  function info(p) {
    var traces = [{type: "scatter", mode: "markers", name: "effective battles",
      x: p.points.map(function (q) { return q.rank; }), y: p.points.map(function (q) { return q.effective; }),
      text: p.points.map(function (q) { return plain(q.name); }), marker: {color: "#9db4c0"}}];
    var shapes = [];
    var extra = {xaxis: {title: {text: "Headline rank"}}, yaxis: {title: {text: "Effective battles"}}};
    var shrunk = p.points.filter(function (q) { return q.contraction !== null; });
    if (shrunk.length > 0) {
      traces.push({type: "scatter", mode: "markers", name: "skill contraction", yaxis: "y2",
        x: shrunk.map(function (q) { return q.rank; }), y: shrunk.map(function (q) { return q.contraction; }),
        text: shrunk.map(function (q) { return plain(q.name); }), marker: {color: "#D55E00"}});
      shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: p.line, y1: p.line,
        line: {color: "#D55E00", dash: "dash", width: 1}});
      extra.yaxis2 = {title: {text: "Skill contraction 1 - (sd/sigma)^2"}, overlaying: "y", side: "right", showgrid: false};
    }
    extra.shapes = shapes;
    plot("info_vs_rank", traces, extra);
  }

  function troops(p) {
    var byEra = {};
    p.eras.forEach(function (e) { byEra[e.era] = e; });
    function share(key) {
      return p.order.map(function (era) {
        var e = byEra[era];
        var total = e.multi + e.single + e.imputed + (e.other || 0) + (e.bound || 0) || 1;
        return e[key] / total;
      });
    }
    var prov = [
      {type: "bar", name: "multi-source", x: p.order, y: share("multi"), marker: {color: "#0072B2"}},
      {type: "bar", name: "single report", x: p.order, y: share("single"), marker: {color: "#E69F00"}},
      {type: "bar", name: "imputed", x: p.order, y: share("imputed"), marker: {color: "#D55E00"}}
    ];
    if (p.eras.some(function (e) { return e.other > 0; })) {
      prov.push({type: "bar", name: "other method", x: p.order, y: share("other"), marker: {color: "#999999"}});
    }
    if (p.eras.some(function (e) { return e.bound > 0; })) {
      prov.push({type: "bar", name: "bound only", x: p.order, y: share("bound"), marker: {color: "#CC79A7"}});
    }
    plot("troop_provenance", prov, {barmode: "stack", yaxis: {title: {text: "Share of sides"}}});
    var sc = document.getElementById("chart-cov-troop_scatter");
    if (!p.scatter || p.scatter.length === 0) { sc.textContent = ""; return; }
    var eras = Object.keys(ERA).filter(function (e) { return p.scatter.some(function (q) { return q.era === e; }); });
    var traces = eras.map(function (e) {
      var rows = p.scatter.filter(function (q) { return q.era === e; });
      return {type: "scatter", mode: "markers", name: e, x: rows.map(function (q) { return q.reported; }),
        y: rows.map(function (q) { return q.reconciled; }), marker: {color: ERA[e], size: 5}};
    });
    var lo = Math.min.apply(null, p.scatter.map(function (q) { return Math.min(q.reported, q.reconciled); }));
    var hi = Math.max.apply(null, p.scatter.map(function (q) { return Math.max(q.reported, q.reconciled); }));
    traces.push({type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], showlegend: false, hoverinfo: "skip",
      line: {color: css("--muted"), dash: "dash"}});
    Plotly.newPlot(sc, traces, layout({xaxis: {title: {text: "Mean reported total"}, type: "log"},
      yaxis: {title: {text: "Reconciled total"}, type: "log"}}), {responsive: true});
  }

  function fame(p) {
    plot("fame_bias", [{type: "bar", x: p.bins.map(function (b) { return b.label; }),
      y: p.bins.map(function (b) { return b.rate; }), text: p.bins.map(function (b) { return "n=" + b.n; }),
      textposition: "outside", marker: {color: "#9db4c0"}}],
      {xaxis: {title: {text: "Credited battles of the general"}, type: "category"},
       yaxis: {title: {text: "Win rate of top commanders"}, range: [0, 1.1]}});
  }

  var DRAW = {timeline: timeline, components: components, credit_source: creditSource,
    info_vs_rank: info, troop_provenance: troops, fame_bias: fame};

  function drawAll() {
    Object.keys(DRAW).forEach(function (key) {
      if (!cov[key]) { missing(key); return; }
      if (typeof Plotly === "undefined") { host(key).textContent = "Charts need plotly.js."; return; }
      DRAW[key](cov[key]);
    });
    drawn = true;
  }

  function resizeAll() {
    if (typeof Plotly === "undefined" || !Plotly.Plots) { return; }
    Object.keys(DRAW).concat(["troop_scatter"]).forEach(function (key) {
      var el = host(key);
      if (el && el.data) { Plotly.Plots.resize(el); }
    });
  }

  function show(which) {
    var isCov = which === "coverage";
    document.getElementById("tab-ranking").hidden = isCov;
    document.getElementById("tab-coverage").hidden = !isCov;
    document.getElementById("tabbtn-ranking").setAttribute("aria-selected", String(!isCov));
    document.getElementById("tabbtn-coverage").setAttribute("aria-selected", String(isCov));
    if (isCov && !drawn) { drawAll(); } else if (isCov) { resizeAll(); }
    if (!isCov && window.resizeRankingCharts) { window.resizeRankingCharts(); }
  }

  document.getElementById("tabbtn-ranking").addEventListener("click", function () { show("ranking"); });
  document.getElementById("tabbtn-coverage").addEventListener("click", function () { show("coverage"); });
  if (window.matchMedia) {
    var scheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (scheme.addEventListener) { scheme.addEventListener("change", function () { if (drawn) { drawAll(); } }); }
  }
})();
"""
