"""
Live HTML report for long-running jobs.

One self-describing JSON document (status, named widgets, their points) rendered into a linear
Tressoir page with Plotly.js, rewritten atomically as the job progresses. Every
number the page shows is in the embedded data block, mirrored to report_data.json beside it. The
page renders inside the Tressoir VS Code webview (which injects the markup after load, morphs it in
place on file changes and fires `tressoir:render`) and in a plain browser (timed reload).

`HtmlReporter` is job-agnostic: widgets are created by name (line plots, bar plots, tables, text
blocks) and fed through `add_data_point`. Job-specific reporters subclass it and own the widget
layout, e.g. `retrieval.RetrievalReporter`.

The page is self-contained: the data is embedded, and the Tressoir linear page assets, CodeMirror
and Plotly come from pinned HTTPS URLs (jsDelivr, cdnjs, cdn.plot.ly), so one file works in the
VS Code renderer and in a browser with nothing beside it. No assets are copied next to the report.
"""
import html
import json
import os
import tempfile
import time
from pathlib import Path

REPORT_FILENAME = "report.tressoir.html"
DATA_FILENAME = "report_data.json"

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="__DESCRIPTION__">
  <title>__TITLE__</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/amlatyrngom/tressoir-external@v0.1.7/extension/src/notebook/assets/linear/tressoir-linear.css">
  <style>
    .report-status {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
      gap: var(--space-2) var(--space-3);
      margin-block: var(--space-3);
      padding: var(--space-3);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
    }
    .report-status div { min-width: 0; }
    .report-status dt { margin: 0; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .report-status dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
    .report-plot { width: 100%; height: 22rem; margin-block: var(--space-2) var(--space-3); }
    .report-plot.tall { height: 26rem; }
    .report-footer { color: var(--muted); font-size: 0.85rem; }
    .report-table td { font-variant-numeric: tabular-nums; white-space: nowrap; }
  </style>
</head>
<body>
  <main class="tressoir-document">
    <header class="document-header">
      <p class="eyebrow">__EYEBROW__ · __STATE__</p>
      <h1 id="report-title">__TITLE__</h1>
      <p class="lede" id="report-description">__DESCRIPTION__</p>
      <ul class="meta" aria-label="Report metadata" id="report-meta"></ul>
    </header>

    <section class="section" aria-labelledby="status-heading">
      <h2 id="status-heading">Status</h2>
      <dl class="report-status" id="report-status"></dl>
    </section>

    <div id="report-widgets"></div>

    <p class="report-footer" id="report-footer"></p>
    <pre id="report-data" hidden style="display:none">__DATA__</pre>

    <aside class="feedback-dock" data-feedback-dock>
      <section class="feedback-popover" id="artifact-feedback-panel" data-feedback-panel role="dialog" aria-modal="false" aria-labelledby="artifact-feedback-title" hidden>
        <header class="feedback-popover-header">
          <h2 id="artifact-feedback-title">Feedback Form</h2>
          <button class="feedback-close" type="button" data-feedback-close aria-label="Close feedback form">×</button>
        </header>
        <div class="feedback-editor">
          <textarea id="artifact-feedback" data-tressoir-feedback aria-label="Feedback Form (Markdown)"></textarea>
        </div>
      </section>
      <button class="feedback-trigger" type="button" data-feedback-toggle aria-expanded="false" aria-controls="artifact-feedback-panel" aria-label="Open feedback form" title="Feedback">
        <svg data-tressoir-inline viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 5.75h14v10.5H9l-4 3v-13.5Z"></path>
          <path d="M8.25 9h7.5M8.25 12.5h5"></path>
        </svg>
      </button>
    </aside>
  </main>

  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/xml/xml.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/meta.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/markdown/markdown.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/gh/amlatyrngom/tressoir-external@v0.1.7/extension/src/notebook/assets/linear/tressoir-linear.js"></script>
  <script defer src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js"></script>
  <script>
  (function () {
    var report = null;
    var plots = [];

    function cssColor(token, fallback) {
      var probe = document.createElement("span");
      probe.style.color = "var(" + token + ", " + fallback + ")";
      document.getElementById("report-widgets").appendChild(probe);
      var value = getComputedStyle(probe).color;
      probe.remove();
      return value || fallback;
    }
    function theme() {
      var ink = cssColor("--ink", "#202124"), muted = cssColor("--muted", "#62666d"), line = cssColor("--line", "#d8dadd");
      var palette = [cssColor("--accent", "#176b87"), cssColor("--danger", "#a33a3a"), cssColor("--positive", "#2f7657"),
                     cssColor("--warning", "#946115"), "#7b5ea7", "#4c8fb0"];
      return { ink: ink, muted: muted, line: line, palette: palette,
               font: getComputedStyle(document.body).fontFamily };
    }
    function movingAverage(values, window) {
      var out = [], sum = 0, queue = [];
      for (var i = 0; i < values.length; i++) {
        var v = values[i];
        if (v === null || v === undefined) { out.push(null); continue; }
        queue.push(v); sum += v;
        if (queue.length > window) sum -= queue.shift();
        out.push(sum / queue.length);
      }
      return out;
    }
    function traces(widget, t) {
      var out = [];
      widget.series.forEach(function (series, i) {
        var color = t.palette[i % t.palette.length];
        var many = series.x.length > 200;
        if (widget.type === "bar") {
          out.push({ type: "bar", name: series.name, x: series.x, y: series.y, marker: { color: color } });
          return;
        }
        var smooth = widget.smoothing_window || 0;
        out.push({
          type: "scatter", name: series.name, x: series.x, y: series.y,
          mode: many ? "lines" : "lines+markers",
          line: { color: color, width: smooth ? 1 : 2 },
          marker: { color: color, size: 5 },
          opacity: smooth ? 0.35 : 1,
          hoverinfo: smooth ? "skip" : undefined,
          showlegend: !smooth,
        });
        if (smooth) {
          out.push({
            type: "scatter", name: series.name + " (mean of " + smooth + ")", x: series.x,
            y: movingAverage(series.y, smooth), mode: "lines", line: { color: color, width: 2 },
          });
        }
      });
      return out;
    }
    function layout(widget, t) {
      var l = {
        margin: { l: 60, r: 20, t: 10, b: 50 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: t.ink, family: t.font, size: 12 },
        xaxis: { title: { text: widget.x_label }, gridcolor: t.line, zerolinecolor: t.line, linecolor: t.line, automargin: true },
        yaxis: { title: { text: widget.y_label }, gridcolor: t.line, zerolinecolor: t.line, linecolor: t.line, automargin: true },
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: t.muted } },
        hovermode: "closest",
        autosize: true,
      };
      if (widget.log_y) { l.yaxis.type = "log"; l.yaxis.exponentformat = "e"; l.yaxis.showexponent = "all"; }
      if (widget.type === "bar") l.xaxis.type = "category";
      return l;
    }
    function drawAll() {
      if (!window.Plotly) return;
      var t = theme();
      plots.forEach(function (p) {
        if (!p.div.isConnected) return;
        Plotly.react(p.div, traces(p.widget, t), layout(p.widget, t), { displaylogo: false, responsive: false });
      });
    }
    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }
    function build() {
      report = JSON.parse(document.getElementById("report-data").textContent);
      if (window.Plotly) plots.forEach(function (p) { try { Plotly.purge(p.div); } catch (_) {} });
      plots = [];
      document.title = report.title;
      var meta = document.getElementById("report-meta");
      meta.replaceChildren();
      ["Updated " + report.updated_at, "Refreshes every " + report.refresh_seconds + " s", "Self-contained: data embedded, libraries from pinned HTTPS URLs"]
        .forEach(function (text) { meta.appendChild(el("li", "", text)); });
      var status = document.getElementById("report-status");
      status.replaceChildren();
      Object.keys(report.status).forEach(function (key) {
        var item = el("div");
        item.appendChild(el("dt", "", key));
        item.appendChild(el("dd", "", report.status[key]));
        status.appendChild(item);
      });
      var root = document.getElementById("report-widgets");
      root.replaceChildren();
      report.widgets.forEach(function (w) {
        var id = "widget-" + w.name;
        var section = el("section", "section");
        section.setAttribute("aria-labelledby", id);
        var h = el("h2", "", w.title); h.id = id; section.appendChild(h);
        if (w.description) section.appendChild(el("p", "", w.description));
        if (w.type === "line" || w.type === "bar") {
          var div = el("div", "report-plot" + (w.smoothing_window ? " tall" : ""));
          section.appendChild(div);
          plots.push({ div: div, widget: w });
        } else if (w.type === "table") {
          var region = el("div", "scroll-region");
          var table = el("table", "table report-table");
          var head = table.createTHead().insertRow();
          w.columns.forEach(function (c) { head.appendChild(el("th", "", c)); });
          var body = table.createTBody();
          w.rows.forEach(function (row) {
            var tr = body.insertRow();
            w.columns.forEach(function (c) { tr.insertCell().textContent = row[c] === undefined || row[c] === null ? "" : row[c]; });
          });
          region.appendChild(table); section.appendChild(region);
        } else if (w.type === "text") {
          var pre = el("pre", "code-block");
          pre.appendChild(el("code", "", w.text));
          section.appendChild(pre);
        }
        root.appendChild(section);
      });
      document.getElementById("report-footer").textContent =
        "Updated " + report.updated_at + " · refreshes every " + report.refresh_seconds + " s · plots: Plotly.js basic 2.35.2 from cdn.plot.ly · page assets: Tressoir linear v0.1.7 and CodeMirror 5.65.16 from CDNs";
    }
    function render() {
      build();
      var tries = 0;
      (function whenPlotly() {
        if (window.Plotly) return drawAll();
        if (tries++ < 600) setTimeout(whenPlotly, 50);   // up to 30 s for the network
      })();
    }
    function boot() {
      // The Tressoir extension injects this page after load and fires tressoir:render once the
      // scripts have run, and again after it morphs the file in place; a plain browser renders
      // now and gets a timed reload instead.
      document.addEventListener("tressoir:render", render);
      if (!window.tressoirNotebook) render();
      if (/^(https?|file):$/.test(location.protocol) && !window.tressoirNotebook) {
        setTimeout(function () { location.reload(); }, report.refresh_seconds * 1000);
      }
      if (window.matchMedia) {
        var mq = window.matchMedia("(prefers-color-scheme: dark)");
        (mq.addEventListener ? mq.addEventListener("change", drawAll) : mq.addListener(drawAll));
      }
      if (window.MutationObserver) {
        new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme-kind"] });
      }
      window.addEventListener("resize", function () {
        if (window.Plotly) plots.forEach(function (p) { if (p.div.isConnected) Plotly.Plots.resize(p.div); });
      });
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
  })();
  </script>
</body>
</html>
"""


def render_report_page(report: dict) -> str:
    """The HTML the reporter writes. Everything the page shows comes from the embedded JSON."""
    return (
        _PAGE_TEMPLATE
        .replace("__TITLE__", html.escape(report["title"]))
        .replace("__DESCRIPTION__", html.escape(report["description"]))
        .replace("__EYEBROW__", html.escape(report.get("eyebrow", "Report")))
        .replace("__STATE__", "finished" if report.get("finished") else "running")
        .replace("__DATA__", html.escape(json.dumps(report), quote=False))
    )


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def format_rate(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 10_000 else f"{value:.1f}"


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.chmod(tmp, 0o644)                                                              # mkstemp gives 600; reports are shared files
    os.replace(tmp, path)


class HtmlReporter:
    """
    Reports progress in a continually updated HTML file.
    Widgets are created by name with their axes or columns and fed through add_data_point.
    """
    def __init__(
        self,
        folder: str,
        title: str,
        description: str,
        eyebrow: str = "Report",
        refresh_seconds: int = 30,
        min_render_interval_seconds: float = 5.0,
    ):
        self.folder = Path(folder)
        self.title = title
        self.description = description
        self.eyebrow = eyebrow
        self.refresh_seconds = refresh_seconds
        self.min_render_interval_seconds = min_render_interval_seconds
        self.status: dict[str, str] = {}
        self.widgets: dict[str, dict] = {}
        self.finished = False
        self.last_render_time = 0.0

    # ----------------------------------------------------------------------------- widgets
    def set_status(self, **fields: str) -> None:
        """The status strip; replaces previous values of the given keys."""
        for key, value in fields.items():
            self.status[key.replace("_", " ")] = str(value)

    def _add_widget(self, name: str, widget: dict) -> None:
        assert name not in self.widgets, f"Widget {name!r} already exists."
        self.widgets[name] = widget

    def initialize_line_plot(
        self, name: str, title: str, description: str, x_label: str, y_label: str,
        series: list[str], log_y: bool = False, smoothing_window: int = 0,
    ) -> None:
        self._add_widget(name, {
            "type": "line", "name": name, "title": title, "description": description,
            "x_label": x_label, "y_label": y_label, "log_y": log_y, "smoothing_window": smoothing_window,
            "series": [{"name": series_name, "x": [], "y": []} for series_name in series],
        })

    def initialize_bar_plot(
        self, name: str, title: str, description: str, x_label: str, y_label: str,
        series: list[str] | None = None,
    ) -> None:
        self._add_widget(name, {
            "type": "bar", "name": name, "title": title, "description": description,
            "x_label": x_label, "y_label": y_label,
            "series": [{"name": series_name, "x": [], "y": []} for series_name in (series or ["value"])],
        })

    def initialize_table(self, name: str, title: str, description: str, columns: list[str]) -> None:
        self._add_widget(name, {
            "type": "table", "name": name, "title": title, "description": description,
            "columns": list(columns), "rows": [],
        })

    def set_text(self, name: str, title: str, text: str) -> None:
        """A text block; calling again with the same name replaces it."""
        self.widgets[name] = {"type": "text", "name": name, "title": title, "description": "", "text": text}

    def add_data_point(self, name: str, data: dict) -> None:
        """
        line: {"x": float, "<series>": float, ...}; bar: {"label": str, "<series>": float} or {"label", "value"};
        table: {column: value} (a row keyed "running" is replaced, not appended). Unknown names or keys raise.
        """
        assert name in self.widgets, f"Unknown widget {name!r}."
        widget = self.widgets[name]
        if widget["type"] in ("line", "bar"):
            key = "x" if widget["type"] == "line" else "label"
            assert key in data, f"Widget {name!r} needs {key!r} in every data point."
            by_name = {series["name"]: series for series in widget["series"]}
            for series_name, value in data.items():
                if series_name == key:
                    continue
                assert series_name in by_name, f"Widget {name!r} has no series {series_name!r}."
                by_name[series_name]["x"].append(data[key])
                by_name[series_name]["y"].append(value)
        elif widget["type"] == "table":
            unknown = set(data) - set(widget["columns"]) - {"running"}
            assert not unknown, f"Widget {name!r} has no columns {sorted(unknown)}."
            row = {column: data.get(column, "") for column in widget["columns"]}
            rows = widget["rows"]
            if rows and rows[-1].get("running"):
                rows.pop()
            if data.get("running"):
                row["running"] = True
            rows.append(row)
        else:
            raise AssertionError(f"Widget {name!r} is a text block; use set_text.")

    # ----------------------------------------------------------------------------- rendering
    def document(self) -> dict:
        widgets = []
        for widget in self.widgets.values():
            widget = dict(widget)
            if widget["type"] == "table":
                widget["rows"] = [{key: value for key, value in row.items() if key != "running"} for row in widget["rows"]]
            widgets.append(widget)
        return {
            "title": self.title,
            "description": self.description,
            "eyebrow": self.eyebrow,
            "refresh_seconds": self.refresh_seconds,
            "finished": self.finished,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": dict(self.status),
            "widgets": widgets,
        }

    def render(self, force: bool = False) -> None:
        """Write report.tressoir.html and report_data.json atomically; throttled unless forced."""
        now = time.time()
        if not force and now - self.last_render_time < self.min_render_interval_seconds:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        report = self.document()
        _write_atomic(self.folder / DATA_FILENAME, json.dumps(report, indent=1))
        _write_atomic(self.folder / REPORT_FILENAME, render_report_page(report))
        self.last_render_time = now

    def finish(self) -> None:
        """Mark the run finished and render one last time."""
        self.finished = True
        self.render(force=True)
