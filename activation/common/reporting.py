"""
Live HTML report for long-running jobs.

One self-describing JSON document (status, named widgets, their points) rendered into a linear
Tressoir page with Plotly.js. `report_data.json` is rewritten atomically on every render and is the
live path: the page polls it (every `poll_seconds`) and updates its existing DOM in place, keeping
sections by widget name, redrawing plots with Plotly.react (the viewer's zoom and legend state
survive), rebuilding only table rows and changing text only when it changed. The HTML file itself is
written on the first render, whenever a widget is added, removed or reordered, when the run finishes,
and otherwise at most every `page_rewrite_seconds`. Every number the page shows is also in the
embedded data block, which is what the page shows when the poll cannot fetch (a file:// page, or a
viewer without the folder as a resource root): the failure is stated in the page's metadata line and
the fetch is retried with a bounded backoff; the page never reloads itself. Inside the Tressoir
VS Code webview (which injects the markup after load, morphs it in place on file changes and fires
`tressoir:render`) repeated render events are harmless: they adopt the embedded data only when it is
newer than what the page shows. There is one polling loop and one boot per page.

`HtmlReporter` is job-agnostic: widgets are created by name (line plots, bar plots, tables, text
blocks, rendered trajectories) and fed through `add_data_point`. Job-specific reporters subclass it and own the widget
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
    .report-live-failure { color: var(--warning); }
    .report-table td { font-variant-numeric: tabular-nums; white-space: nowrap; }
    .trajectory { border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); padding: var(--space-2) var(--space-3); margin-block: var(--space-2); }
    .trajectory > summary { cursor: pointer; font-weight: 600; }
    .trajectory > summary .traj-stats { color: var(--muted); font-weight: 400; font-size: 0.85rem; margin-left: 0.5rem; }
    .trajectory > summary .traj-state { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; margin-left: 0.5rem; color: var(--accent); }
    .trajectory > summary .traj-state.finished { color: var(--muted); }
    .traj-step { margin-block: var(--space-2); padding-left: var(--space-3); border-left: 3px solid var(--line); }
    .traj-step.assistant { border-color: var(--accent); }
    .traj-step.tool { border-color: var(--positive); }
    .traj-step.prompt { border-color: var(--warning); }
    .traj-role { display: block; color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .traj-step p { white-space: pre-wrap; margin-block: 0.25rem; }
    .traj-step pre { max-height: 20rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin-block: 0.25rem; }
    .traj-call { color: var(--muted); font-size: 0.85rem; margin-block: 0.25rem 0; }
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
    // The page keeps one DOM and updates it in place: sections are kept by widget name, plots are
    // redrawn with Plotly.react (zoom and legend state survive), tables rebuild only their rows,
    // text blocks change only when their text changed. There is one polling loop and one boot;
    // repeated tressoir:render events are harmless; a failed fetch is shown, never a reload.
    if (window.__reportPageBooted) return;
    window.__reportPageBooted = true;
    var report = null;
    var plots = {};            // widget name -> { div, widget }
    var sections = {};         // widget name -> <section>
    var polling = false;
    var pollTimer = null;
    var pollDelay = 0;
    var fetchFailure = null;   // { at, reason } while live data is unavailable
    var embedded = JSON.parse(document.getElementById("report-data").textContent);

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
        uirevision: widget.name,   // a constant per widget: Plotly.react keeps the viewer's zoom and legend state
      };
      if (widget.log_y) { l.yaxis.type = "log"; l.yaxis.exponentformat = "e"; l.yaxis.showexponent = "all"; }
      if (widget.type === "bar") l.xaxis.type = "category";
      return l;
    }
    function drawAll() {
      if (!window.Plotly) return;
      var t = theme();
      Object.keys(plots).forEach(function (name) {
        var p = plots[name];
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
    function setText(node, text) {
      if (node.textContent !== text) node.textContent = text;
    }
    function codeBlock(text) {
      var pre = el("pre", "code-block");
      pre.appendChild(el("code", "", text));
      return pre;
    }
    function trajectory(item) {
      // One rollout: a collapsible block, open while running, with the prompt then every step.
      var details = el("details", "trajectory");
      if (item.state === "running") details.open = true;
      var summary = el("summary", "", item.title);
      summary.appendChild(el("span", "traj-state " + item.state, item.state));
      summary.appendChild(el("span", "traj-stats", item.stats));
      if (item.file) {
        var link = el("a", "traj-stats", "full trajectory");
        link.href = item.file;
        summary.appendChild(link);
      }
      details.appendChild(summary);
      if (item.prompt) {
        var prompt = el("div", "traj-step prompt");
        prompt.appendChild(el("span", "traj-role", "prompt"));
        prompt.appendChild(codeBlock(item.prompt));
        details.appendChild(prompt);
      }
      item.steps.forEach(function (step) {
        var div = el("div", "traj-step " + step.role);
        div.appendChild(el("span", "traj-role", step.role + (step.turn ? " · turn " + step.turn : "")));
        if (step.content) div.appendChild(el("p", "", step.content));
        (step.calls || []).forEach(function (call) {
          div.appendChild(el("div", "traj-call", "→ " + call.name));
          if (call.arguments) div.appendChild(codeBlock(call.arguments));
        });
        (step.results || []).forEach(function (result) {
          div.appendChild(el("div", "traj-call", "← " + result.name));
          div.appendChild(codeBlock(result.output));
        });
        details.appendChild(div);
      });
      return details;
    }
    // Reorder `parent`'s children to match `wanted` (a list of nodes), moving only nodes that are out of place.
    function reorder(parent, wanted) {
      var cursor = parent.firstElementChild;
      for (var i = 0; i < wanted.length; i++) {
        var node = wanted[i];
        if (cursor === node) { cursor = cursor.nextElementSibling; continue; }
        parent.insertBefore(node, cursor);
      }
    }
    function updateMeta() {
      var meta = document.getElementById("report-meta");
      var lines = ["Updated " + report.updated_at,
                   (polling ? "Polling report_data.json every " + report.poll_seconds + " s" : "Showing the data embedded in the page"),
                   "Self-contained: data embedded, libraries from pinned HTTPS URLs"];
      if (fetchFailure) {
        lines.push("Live data unavailable since " + fetchFailure.at + " (" + fetchFailure.reason + "); showing the last data this page has; retrying");
      }
      while (meta.children.length > lines.length) meta.removeChild(meta.lastChild);
      lines.forEach(function (text, i) {
        var li = meta.children[i];
        if (!li) { li = el("li"); meta.appendChild(li); }
        if (i === lines.length - 1 && fetchFailure) li.className = "report-live-failure"; else if (li.className) li.className = "";
        setText(li, text);
      });
    }
    function updateStatus() {
      var status = document.getElementById("report-status");
      var keys = Object.keys(report.status);
      var byKey = {};
      Array.prototype.forEach.call(status.children, function (item) { byKey[item.getAttribute("data-key")] = item; });
      var wanted = keys.map(function (key) {
        var item = byKey[key];
        if (!item) {
          item = el("div"); item.setAttribute("data-key", key);
          item.appendChild(el("dt", "", key)); item.appendChild(el("dd", "", ""));
          status.appendChild(item);
        }
        setText(item.lastChild, report.status[key]);
        delete byKey[key];
        return item;
      });
      Object.keys(byKey).forEach(function (key) { byKey[key].remove(); });
      reorder(status, wanted);
    }
    function updateTable(section, w) {
      var table = section.querySelector("table");
      var columnsKey = JSON.stringify(w.columns);
      if (!table || table.getAttribute("data-columns") !== columnsKey) {
        var region = section.querySelector(".scroll-region");
        if (region) region.remove();
        region = el("div", "scroll-region");
        table = el("table", "table report-table");
        table.setAttribute("data-columns", columnsKey);
        var head = table.createTHead().insertRow();
        w.columns.forEach(function (c) { head.appendChild(el("th", "", c)); });
        table.createTBody();
        region.appendChild(table); section.appendChild(region);
      }
      var body = table.tBodies[0];
      var rowsKey = JSON.stringify(w.rows);
      if (body.getAttribute("data-rows") === rowsKey) return;
      body.setAttribute("data-rows", rowsKey);
      body.replaceChildren();
      w.rows.forEach(function (row) {
        var tr = body.insertRow();
        w.columns.forEach(function (c) { tr.insertCell().textContent = row[c] === undefined || row[c] === null ? "" : row[c]; });
      });
    }
    function updateTrajectories(section, w) {
      var key = JSON.stringify(w.items);
      if (section.getAttribute("data-items") === key) return;
      section.setAttribute("data-items", key);
      Array.prototype.slice.call(section.children).forEach(function (child) {
        if (child.tagName !== "H2" && !child.classList.contains("widget-description")) child.remove();
      });
      w.items.forEach(function (item) { section.appendChild(trajectory(item)); });
      if (!w.items.length) section.appendChild(el("p", "", "No trajectories yet."));
    }
    function updateWidget(w) {
      var id = "widget-" + w.name;
      var section = sections[w.name];
      if (!section) {
        section = el("section", "section");
        section.id = id;
        section.setAttribute("aria-labelledby", id + "-heading");
        var h = el("h2", "", w.title); h.id = id + "-heading"; section.appendChild(h);
        section.appendChild(el("p", "widget-description", ""));
        sections[w.name] = section;
        document.getElementById("report-widgets").appendChild(section);
      }
      setText(section.firstElementChild, w.title);
      var description = section.querySelector(".widget-description");
      setText(description, w.description || "");
      description.hidden = !w.description;
      if (w.type === "line" || w.type === "bar") {
        var p = plots[w.name];
        if (!p) {
          var div = el("div", "report-plot" + (w.smoothing_window ? " tall" : ""));
          section.appendChild(div);
          p = plots[w.name] = { div: div, widget: w };
        }
        p.widget = w;
      } else if (w.type === "table") {
        updateTable(section, w);
      } else if (w.type === "text") {
        var pre = section.querySelector("pre.code-block");
        if (!pre) { pre = codeBlock(""); section.appendChild(pre); }
        setText(pre.firstChild, w.text);
      } else if (w.type === "trajectories") {
        updateTrajectories(section, w);
      }
      return section;
    }
    function update() {
      if (!report) report = embedded;
      if (document.title !== report.title) document.title = report.title;
      setText(document.getElementById("report-title"), report.title);
      setText(document.getElementById("report-description"), report.description);
      updateMeta();
      updateStatus();
      var root = document.getElementById("report-widgets");
      var seen = {};
      var wanted = report.widgets.map(function (w) { seen[w.name] = true; return updateWidget(w); });
      Object.keys(sections).forEach(function (name) {
        if (seen[name]) return;
        var gone = plots[name];
        if (gone && window.Plotly) { try { Plotly.purge(gone.div); } catch (_) {} }
        delete plots[name];
        sections[name].remove();
        delete sections[name];
      });
      reorder(root, wanted);
      setText(document.getElementById("report-footer"),
        "Updated " + report.updated_at + " · plots: Plotly.js basic 2.35.2 from cdn.plot.ly · page assets: Tressoir linear v0.1.7 and CodeMirror 5.65.16 from CDNs");
    }
    function render() {
      update();
      var tries = 0;
      (function whenPlotly() {
        if (window.Plotly) return drawAll();
        if (tries++ < 600) setTimeout(whenPlotly, 50);   // up to 30 s for the network
      })();
    }
    function schedulePoll(seconds) {
      if (pollTimer !== null) clearTimeout(pollTimer);
      pollTimer = setTimeout(poll, seconds * 1000);
    }
    function poll() {
      // report_data.json beside the page is rewritten on every render; the page itself only rarely.
      if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null; }
      if (report && report.finished && polling) return;
      var base = (report && report.poll_seconds) || 2;
      fetch("report_data.json?t=" + Date.now(), { cache: "no-store" }).then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      }).then(function (fresh) {
        var recovered = fetchFailure !== null || !polling;
        polling = true; fetchFailure = null; pollDelay = 0;
        if (JSON.stringify(fresh) !== JSON.stringify(report)) { report = fresh; render(); }
        else if (recovered) updateMeta();
        schedulePoll(report.poll_seconds || 2);
      }).catch(function (error) {
        // No reload: the data already on the page stays, the failure is shown, and the fetch is
        // retried with a bounded backoff (the extension delivers file changes through
        // tressoir:render in the meantime).
        if (!fetchFailure) fetchFailure = { at: new Date().toLocaleTimeString(), reason: (error && error.message) || String(error) };
        pollDelay = Math.min(60, pollDelay ? pollDelay * 2 : base);
        updateMeta();
        schedulePoll(pollDelay);
      });
    }
    function onRender() {
      // Fired by the Tressoir extension after it injects the page and after each in-place morph:
      // adopt the embedded data only when it is newer than what the page already shows.
      var fresh = JSON.parse(document.getElementById("report-data").textContent);
      if (!report || fresh.updated_at > report.updated_at) report = fresh;
      render();
    }
    function boot() {
      // The Tressoir extension injects this page after load and fires tressoir:render once the
      // scripts have run, and again after it morphs the file in place; a plain browser renders
      // now. Either way the page then polls report_data.json. Everything here runs once.
      document.addEventListener("tressoir:render", onRender);
      if (!window.tressoirNotebook) render();
      poll();
      if (window.matchMedia) {
        var mq = window.matchMedia("(prefers-color-scheme: dark)");
        (mq.addEventListener ? mq.addEventListener("change", drawAll) : mq.addListener(drawAll));
      }
      if (window.MutationObserver) {
        new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme-kind"] });
      }
      window.addEventListener("resize", function () {
        if (window.Plotly) Object.keys(plots).forEach(function (name) { var p = plots[name]; if (p.div.isConnected) Plotly.Plots.resize(p.div); });
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
        poll_seconds: float = 2.0,
        page_rewrite_seconds: float = 60.0,
    ):
        self.folder = Path(folder)
        self.title = title
        self.description = description
        self.eyebrow = eyebrow
        self.refresh_seconds = refresh_seconds
        self.min_render_interval_seconds = min_render_interval_seconds
        self.status: dict[str, str] = {}
        self.widgets: dict[str, dict] = {}
        self.poll_seconds = poll_seconds
        self.page_rewrite_seconds = page_rewrite_seconds
        self.finished = False
        self.last_render_time = 0.0
        self.last_page_time = 0.0
        self.last_page_widgets: list[str] | None = None     # widget names as of the last page write

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

    def set_trajectories(self, name: str, title: str, description: str, items: list[dict]) -> None:
        """
        Rendered agent trajectories; calling again replaces the list. Each item: {"title", "state"
        ("running" | "finished"), "stats", "prompt", "file" (relative link or None), "steps": [{"role", "turn", "content", "calls":
        [{"name", "arguments"}], "results": [{"name", "output"}]}]} with every text already cut to size.
        """
        self.widgets[name] = {"type": "trajectories", "name": name, "title": title, "description": description, "items": items}

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
            raise AssertionError(f"Widget {name!r} is a {widget['type']} block; use set_text / set_trajectories.")

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
            "poll_seconds": self.poll_seconds,
            "finished": self.finished,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": dict(self.status),
            "widgets": widgets,
        }

    def remove_widget(self, name: str) -> None:
        self.widgets.pop(name, None)

    def render(self, force: bool = False) -> None:
        """
        Write report_data.json atomically (throttled unless forced): that file is the live path the
        page polls. The page itself is written on the first render, whenever the set or order of
        widgets changed since it was last written, when the run finishes, and otherwise at most every
        page_rewrite_seconds so a page opened without polling is not too stale.
        """
        now = time.time()
        if not force and now - self.last_render_time < self.min_render_interval_seconds:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        report = self.document()
        _write_atomic(self.folder / DATA_FILENAME, json.dumps(report, indent=1))
        page = self.folder / REPORT_FILENAME
        widget_names = [widget["name"] for widget in report["widgets"]]
        structure_changed = widget_names != self.last_page_widgets
        if self.finished or not page.exists() or structure_changed or now - self.last_page_time >= self.page_rewrite_seconds:
            _write_atomic(page, render_report_page(report))
            self.last_page_time = now
            self.last_page_widgets = widget_names
        self.last_render_time = now

    def finish(self) -> None:
        """Mark the run finished and render one last time."""
        self.finished = True
        self.render(force=True)
