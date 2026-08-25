/* tressoir-md.js — USER_ARTIFACT_MD core runtime (classic IIFE, eval-free).
 *
 * Projects a `.tressoir.md` source into the LOCKED user-artifact DOM (artifact.css) and wires
 * the runtime behaviors. Loaded as a classic <script src> alongside the vendored UMD libs;
 * consumed by the bridge webview (which calls TressoirMd.project() then morphs) and by the
 * standalone SHIM (TressoirMd.render()).
 *
 * Public authoring is ordinary Markdown: a leading H1 + lede, visible sections, and exact shared
 * HTML card/decision components only when interaction is useful. Legacy card/item/input directives
 * remain accepted and are adapted into the current component DOM.
 *
 * Dependencies (window globals, all eval-free UMD): TressoirRemark (unified/remark/rehype
 * bundle), jsyaml, Prism, CodeMirror. No `eval`/`new Function`; projection is a mdast->HTML
 * string transform + classic DOM wiring.
 */
(function (global) {
  'use strict';

  // ----------------------------------------------------------------- helpers
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function escAttr(s) {
    return esc(s).replace(/"/g, '&quot;');
  }
  function feedbackEnabled(fm) {
    if (fm && fm.feedback === false) return false;
    var value = fm && fm.feedback;
    return !/^(?:disable|disabled|off|false|none)$/i.test(String(value == null ? '' : value).trim());
  }
  function textOf(node) {
    if (!node) return '';
    if (node.type === 'text' || node.type === 'inlineCode') return node.value || '';
    if (node.children) return node.children.map(textOf).join('');
    return '';
  }

  // ----------------------------------------------------------- projection
  function getRemark() {
    var R = global.TressoirRemark;
    if (!R) throw new Error('TressoirRemark (remark UMD bundle) is not loaded.');
    return R;
  }

  // mdast (array of nodes) -> HTML string, converting fenced code blocks to the locked
  // pre.code / pre.diff form the runtime highlighter expects (data-lang). Raw HTML/SVG/scripts
  // pass through via rehype-raw. Markdown headings (### / ####) pass through to <h3>/<h4>,
  // styled by the appended depth rules in artifact.css.
  function renderCodeBlock(node) {
    var lang = String(node.lang || '').toLowerCase();
    var aliases = { py: 'python', rs: 'rust', ts: 'typescript', js: 'javascript', sh: 'bash', shell: 'bash', zsh: 'bash', yml: 'yaml', md: 'markdown' };
    lang = aliases[lang] || lang;
    var classes = 'code-block';
    var language = lang || 'text';
    if (language === 'diff' || language.indexOf('diff-') === 0) classes += ' diff-highlight';
    return '<pre class="' + classes + '"><code class="language-' + escAttr(language) + '">' + esc(node.value) + '</code></pre>';
  }
  function transformCodeDeep(node) {
    if (!node || typeof node !== 'object') return node;
    if (node.type === 'code') {
      return { type: 'html', value: renderCodeBlock(node) };
    }
    if (node.children) {
      node.children = node.children.map(transformCodeDeep);
    }
    return node;
  }
  function mdastToHtml(nodes) {
    var R = getRemark();
    var proc = R.unified()
      .use(R.remarkGfm)
      .use(R.remarkRehype, { allowDangerousHtml: true })
      .use(R.rehypeRaw)
      .use(R.rehypeStringify, { allowDangerousHtml: true });
    var root = { type: 'root', children: nodes.map(transformCodeDeep) };
    var hast = proc.runSync(root);
    return proc.stringify(hast).replace(/<table>/g, '<table class="table">');
  }

  function cardBody(body) {
    return '<div class="card-body"><button class="card-gutter" type="button" data-card-collapse ' +
      'aria-label="Collapse optional section" title="Collapse section"></button>' +
      '<div class="card-content">' + body + '</div></div>';
  }

  // ---- Legacy ::::card -> the canonical native details card.
  // title / oneliner / state are injected RAW so an author can put an inline badge or other
  // inline HTML in them (e.g. state="<span class='badge ok'>Done</span>"). `open` (or
  // `selected`) starts the card expanded. `copy_button=true` opts this card into a control that
  // copies its original Markdown source.
  var _sourceMarkdown = '';
  var _cardMarkdown = [];
  function enabledAttribute(value) {
    if (value === true || value === '') return true;
    return /^(?:true|yes|on|1)$/i.test(String(value == null ? '' : value).trim());
  }
  function renderCard(node) {
    var a = node.attributes || {};
    var open = ('open' in a) || ('selected' in a);
    var title = a.title || '';
    var oneliner = a.oneliner || '';
    var state = a.state || '';
    var copy = '';
    if (enabledAttribute(a.copy_button)) {
      var start = node.position && node.position.start && node.position.start.offset;
      var end = node.position && node.position.end && node.position.end.offset;
      var source = (typeof start === 'number' && typeof end === 'number')
        ? _sourceMarkdown.slice(start, end).trim() : '';
      var copyIndex = _cardMarkdown.push(source) - 1;
      copy = '<button type="button" class="card-copy" data-copy-card="' + copyIndex + '" aria-label="Copy ' + escAttr(title) + ' card as Markdown">Copy Markdown</button>';
    }
    var body = renderBlocks(node.children || []);
    if (copy) body = copy + body;
    return '<details class="card" data-tressoir-markdown' + (open ? ' open' : '') + '>' +
      '<summary><span class="card-title">' + title + '</span>' +
      '<span class="card-oneliner">' + oneliner + '</span>' +
      '<span class="card-badge">' + (state || 'Optional') + '</span></summary>' +
      cardBody(body) + '</details>';
  }

  // ---- :::item{oneliner} -> .ctx-item disclosure row. oneliner injected RAW (inline HTML ok).
  function renderItem(node) {
    var a = node.attributes || {};
    var open = ('open' in a);
    var oneliner = a.oneliner || a.title || '';
    var body = renderBlocks(node.children || []);
    return '<div class="ctx-item' + (open ? ' open' : '') + '" data-morph-keep-class="open">' +
      '<button class="ctx-item-head" type="button" aria-expanded="' + (open ? 'true' : 'false') + '">' +
      '<span class="dot"></span>' + oneliner + '</button>' +
      '<div class="ctx-item-body">' + body + '</div></div>';
  }

  // ---- Legacy :::input -> the canonical checkbox + Free Response decision.
  function renderInput(node) {
    var a = node.attributes || {};
    var key = a.key || ('input.' + Math.random().toString(36).slice(2, 8));
    var children = (node.children || []).slice();
    var question = a.oneliner || a.question || '';
    if (!question && children.length && children[0].type === 'paragraph') {
      question = textOf(children[0]);
      children = children.slice(1);
    }
    if (!question) question = 'Decision';
    var listIndex = children.findIndex(function (child) { return child.type === 'list'; });
    var choices = listIndex >= 0 ? children[listIndex].children || [] : [];
    var choiceHtml = choices.map(function (item, index) {
      var label = textOf(item).trim() || ('Option ' + (index + 1));
      return '<label class="decision-option"><input type="checkbox" data-tressoir-input="' +
        escAttr(key + '.choice.' + (index + 1)) + '"><span><strong>' + esc(label) +
        '</strong></span></label>';
    }).join('');
    if (!choiceHtml) {
      choiceHtml = '<label class="decision-option"><input type="checkbox" data-tressoir-input="' +
        escAttr(key + '.choice.1') + '"><span><strong>Accept it as shown</strong></span></label>';
    }
    var safeId = 'decision-' + String(key).replace(/[^A-Za-z0-9_-]/g, '-');
    return '<article class="decision" data-tressoir-decision data-decision-state="unresolved" ' +
      'data-morph-key="tressoir-input:' + escAttr(key) + '" aria-labelledby="' + safeId + '-question">' +
      '<header class="decision-header"><div><h3 class="decision-title" id="' + safeId + '-question">' + esc(question) +
      '</h3><p class="decision-context">Check every applicable answer. Choosing at least one marks the decision resolved.</p></div>' +
      '<span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span></header>' +
      '<fieldset class="decision-options"><legend class="visually-hidden">Decision answers</legend>' + choiceHtml + '</fieldset>' +
      '<div class="field decision-feedback"><label for="' + safeId + '-response">Free Response</label>' +
      '<textarea id="' + safeId + '-response" rows="2" data-tressoir-input="' + escAttr(key) +
      '" data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea></div></article>';
  }

  // ---- generic block renderer. Legacy items retain their compatibility wrapper.
  function isCardDir(n) { return n.type === 'containerDirective' && n.name === 'card'; }
  function isRowDir(n) { return n.type === 'containerDirective' && (n.name === 'item' || n.name === 'input'); }

  function renderBlocks(nodes) {
    var out = '', i = 0, prose = [];
    function flushProse() { if (prose.length) { out += mdastToHtml(prose); prose = []; } }
    while (i < nodes.length) {
      var n = nodes[i];
      if (isCardDir(n)) {
        flushProse();
        while (i < nodes.length && isCardDir(nodes[i])) { out += renderCard(nodes[i]); i += 1; }
        continue;
      }
      if (isRowDir(n)) {
        flushProse();
        var rows = [];
        while (i < nodes.length && isRowDir(nodes[i])) { rows.push(nodes[i]); i += 1; }
        out += '<div class="ctx-list">' + rows.map(function (it) {
          return it.name === 'input' ? renderInput(it) : renderItem(it);
        }).join('') + '</div>';
        continue;
      }
      prose.push(n); i += 1;
    }
    flushProse();
    return out;
  }

  function renderFeedbackDock(fm) {
    if (!feedbackEnabled(fm)) return '';
    return '<aside class="feedback-dock" data-feedback-dock>' +
      '<section class="feedback-popover" id="artifact-feedback-panel" data-feedback-panel role="dialog" aria-modal="false" aria-labelledby="artifact-feedback-title" hidden>' +
      '<header class="feedback-popover-header"><h2 id="artifact-feedback-title">Feedback Form</h2>' +
      '<button class="feedback-close" type="button" data-feedback-close aria-label="Close feedback form">×</button></header>' +
      '<div class="feedback-editor"><textarea id="artifact-feedback" data-tressoir-feedback aria-label="Feedback Form (Markdown)"></textarea></div></section>' +
      '<button class="feedback-trigger" type="button" data-feedback-toggle aria-expanded="false" aria-controls="artifact-feedback-panel" aria-label="Open feedback form">' +
      '<svg data-tressoir-inline viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5.75h14v9.5H9.5L5 19v-13.25Z"></path><path d="M8.5 9h7M8.5 12h5"></path></svg></button></aside>';
  }

  // CommonMark treats parentheses/brackets as escapable punctuation, so protect the accepted
  // KaTeX delimiters before remark parses them and restore the literal delimiters afterward.
  function protectMath(markdown) {
    var values = [];
    function token(raw) {
      var index = values.push(raw) - 1;
      return 'TRESSOIR_MATH_' + index + '_TOKEN';
    }
    var protectedMarkdown = String(markdown)
      .replace(/\\\[[\s\S]*?\\\]/g, token)
      .replace(/\\\([^\n]*?\\\)/g, token);
    return { markdown: protectedMarkdown, values: values };
  }

  function restoreMath(html, values) {
    values.forEach(function (raw, index) {
      html = html.replace(new RegExp('TRESSOIR_MATH_' + index + '_TOKEN', 'g'), esc(raw));
    });
    return html;
  }

  function extractMarkedCards(markdown) {
    var lines = String(markdown).split(/\r?\n/), output = [], cards = [], findings = [];
    var opener = /^\s*<details\b(?=[^>]*\bclass=(['"])card\1)(?=[^>]*\bdata-tressoir-markdown(?:\s|=|>))[^>]*>\s*$/;
    for (var i = 0; i < lines.length; i += 1) {
      if (!opener.test(lines[i])) { output.push(lines[i]); continue; }
      var start = i, end = -1, nested = false;
      for (var j = i + 1; j < lines.length; j += 1) {
        if (opener.test(lines[j])) nested = true;
        if (/^\s*<\/details>\s*$/.test(lines[j])) { end = j; break; }
      }
      if (nested) findings.push({ level: 'error', line: start + 1, msg: 'marked card blocks cannot nest' });
      if (end < 0) {
        findings.push({ level: 'error', line: start + 1, msg: 'marked card has no matching </details>' });
        output.push(lines[i]);
        continue;
      }
      var inner = lines.slice(i + 1, end).join('\n');
      var summaryStart = inner.indexOf('<summary');
      var summaryEnd = inner.indexOf('</summary>');
      if (summaryStart < 0 || summaryEnd < summaryStart) {
        findings.push({ level: 'error', line: start + 1, msg: 'marked card needs one complete <summary> before its Markdown body' });
        output.push(lines.slice(i, end + 1).join('\n'));
        i = end;
        continue;
      }
      summaryEnd += '</summary>'.length;
      var token = 'TRESSOIR_MARKED_CARD_' + cards.length + '_TOKEN';
      cards.push({
        opener: lines[start].trim(),
        summary: inner.slice(summaryStart, summaryEnd).trim(),
        body: inner.slice(summaryEnd).trim(),
      });
      output.push('', token, '');
      i = end;
    }
    return { markdown: output.join('\n'), cards: cards, findings: findings };
  }

  function renderMarkedCards(html, cards, parseProc) {
    cards.forEach(function (card, index) {
      var bodyTree = parseProc.parse(card.body);
      var rendered = card.opener + card.summary + cardBody(renderBlocks(bodyTree.children || [])) + '</details>';
      html = html.replace('<p>TRESSOIR_MARKED_CARD_' + index + '_TOKEN</p>', rendered);
    });
    return html;
  }

  function project(markdown) {
    var R = getRemark();
    var parseProc = R.unified().use(R.remarkParse).use(R.remarkGfm).use(R.remarkFrontmatter, ['yaml']).use(R.remarkDirective);
    var protectedMath = protectMath(markdown);
    var extracted = extractMarkedCards(protectedMath.markdown);
    var tree = parseProc.parse(extracted.markdown);
    var fm = {};
    var fmNode = null;
    tree.children.forEach(function (n) { if (n.type === 'yaml' && !fmNode) fmNode = n; });
    if (fmNode) { try { fm = global.jsyaml.load(fmNode.value) || {}; } catch (e) { fm = {}; } }
    var content = tree.children.filter(function (n) { return n.type !== 'yaml'; });
    var header = '';
    if (content[0] && content[0].type === 'heading' && content[0].depth === 1) {
      var titleNode = content.shift();
      var ledeNode = content[0] && content[0].type === 'paragraph' ? content.shift() : null;
      header = '<header class="document-header"><h1>' + esc(textOf(titleNode)) + '</h1>' +
        (ledeNode ? mdastToHtml([ledeNode]).replace(/^<p>/, '<p class="lede">') : '') + '</header>';
    } else if (fm.title) {
      header = '<header class="document-header"><h1>' + esc(fm.title) + '</h1>' +
        (fm.description ? '<p class="lede">' + esc(fm.description) + '</p>' : '') + '</header>';
    }
    var preamble = [], sections = [], cur = null;
    content.forEach(function (n) {
      if (n.type === 'heading' && n.depth === 2) { cur = { title: textOf(n), blocks: [] }; sections.push(cur); }
      else if (cur) { cur.blocks.push(n); }
      else { preamble.push(n); }
    });
    _sourceMarkdown = markdown;
    _cardMarkdown = [];
    var body = renderBlocks(preamble);
    sections.forEach(function (s) {
      body += '<section class="section"><h2>' + esc(s.title) + '</h2>' +
        renderBlocks(s.blocks) + '</section>';
    });
    body = renderMarkedCards(body, extracted.cards, parseProc);
    return restoreMath('<main class="tressoir-document">' + header + body + renderFeedbackDock(fm) + '</main>', protectedMath.values);
  }

  // ------------------------------------------------------------- behaviors
  var qsa = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  function adaptTheme() {
    function isDarkBg() {
      try {
        var probe = document.createElement('div');
        probe.style.cssText = 'background:var(--vscode-editor-background, transparent);position:absolute;left:-9999px;';
        document.body.appendChild(probe);
        var c = getComputedStyle(probe).backgroundColor;
        document.body.removeChild(probe);
        var m = c && c.match(/[\d.]+/g);
        if (!m || m.length < 3) return null;
        if (m.length >= 4 && parseFloat(m[3]) === 0) return null;
        var lum = 0.2126 * (+m[0]) + 0.7152 * (+m[1]) + 0.0722 * (+m[2]);
        return lum < 128;
      } catch (_) { return null; }
    }
    var cls = (document.body && document.body.className) || '';
    var dark;
    if (/vscode-dark|vscode-high-contrast(?!-light)/.test(cls)) dark = true;
    else if (/vscode-light|vscode-high-contrast-light/.test(cls)) dark = false;
    else dark = isDarkBg();
    if (dark === null || dark === undefined) return;
    document.documentElement.setAttribute('data-vscode-theme-kind', dark ? 'vscode-dark' : 'vscode-light');
  }

  function readInteraction(key) {
    if (global.tressoirNotebook && typeof global.tressoirNotebook.getInteraction === 'function') {
      try { return global.tressoirNotebook.getInteraction(key); } catch (_) {}
    }
    return undefined;
  }
  function noteInteraction(key, value) {
    if (global.tressoirNotebook && typeof global.tressoirNotebook.storeInteraction === 'function') {
      try { global.tressoirNotebook.storeInteraction(key, value); } catch (_) {}
    }
  }

  function highlightCode() {
    function escTxt(t) { return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    var LANG_ALIASES = { py: 'python', rs: 'rust', ts: 'typescript', tsx: 'tsx', jsx: 'jsx', js: 'javascript',
      sh: 'bash', shell: 'bash', zsh: 'bash', yml: 'yaml', md: 'markdown', 'c++': 'cpp', 'c#': 'csharp', cs: 'csharp', rb: 'ruby', kt: 'kotlin' };
    function grammarFor(lang) {
      if (!global.Prism || !global.Prism.languages || !lang) return null;
      var key = LANG_ALIASES[lang] || lang;
      return global.Prism.languages[key] || null;
    }
    function hl(code, lang) {
      var g = grammarFor(lang);
      if (g && global.Prism) { try { return global.Prism.highlight(code, g, lang); } catch (_) {} }
      return escTxt(code);
    }
    qsa('pre.code').forEach(function (pre) {
      var lang = pre.getAttribute('data-lang') || '';
      pre.innerHTML = hl(pre.textContent.replace(/^\n/, '').replace(/\s+$/, ''), lang);
    });
    qsa('pre.diff').forEach(function (pre) {
      // Idempotent: the diff highlighter joins per-line <span> with '' (dropping the
      // source newlines), so a second pass would collapse everything onto one line.
      // After a real bridge morph the <pre> is reset to raw text (no .dl), so this
      // guard only short-circuits a redundant re-apply on the SAME DOM.
      if (pre.querySelector('.dl')) return;
      var lang = pre.getAttribute('data-lang') || '';
      var lines = pre.textContent.replace(/^\n/, '').replace(/\n$/, '').split('\n');
      pre.innerHTML = lines.map(function (line) {
        if (line.slice(0, 2) === '@@') return '<span class="dl dl-hunk">' + escTxt(line) + '</span>';
        var c = line.charAt(0), cls, marker, bodyTxt;
        if (c === '+') { cls = 'dl-add'; marker = '+'; bodyTxt = line.slice(1); }
        else if (c === '-') { cls = 'dl-del'; marker = '-'; bodyTxt = line.slice(1); }
        else { cls = 'dl-ctx'; marker = ' '; bodyTxt = (c === ' ' ? line.slice(1) : line); }
        return '<span class="dl ' + cls + '"><span class="dm">' + marker + '</span>' + hl(bodyTxt, lang) + '</span>';
      }).join('');
    });
  }

  function applyRuntimeUI() {
    var marker = document.getElementById('gen-marker');
    if (marker) marker.textContent = 'Generated by Tressoir \u2014 ' + new Date().toLocaleString() + '. Consult the user-artifact-md skill before editing.';
    highlightCode();
    // Restore :::input answers: the value is a raw string stored under data-input-key (like
    // free_form_feedback); a non-empty value marks the decision .resolved (green check).
    qsa('textarea[data-input-key]').forEach(function (ta) {
      var key = ta.getAttribute('data-input-key');
      if (!key) return;
      var stored = readInteraction(key);
      var val = stored && (typeof stored === 'object' ? (stored.text != null ? stored.text : stored.value) : stored);
      if (typeof val === 'string') {
        if (ta.value !== val) ta.value = val;
        var dec = ta.closest('.m-decision');
        if (dec && val.length) dec.classList.add('resolved');
      }
    });
    prepareLocalArtifactLinks();
    initFreeForm();
  }

  // CodeMirror's markdown fenced-code highlighting resolves the inner mode via
  // `findModeByName(lang)`, which matches mode NAME/ALIASES only — NOT file extensions.
  // So a fence like ```py (an extension, not an alias) fails to resolve and renders
  // unhighlighted, even though the python mode is loaded. Patch findModeByName to fall
  // back to findModeByExtension so ```py / ```rs / ```ts / ```sh resolve. Idempotent.
  function patchModeResolution() {
    var CM = global.CodeMirror;
    if (!CM || CM.__tressoirModePatched || typeof CM.findModeByName !== 'function') return;
    var origByName = CM.findModeByName;
    CM.findModeByName = function (name) {
      var hit = origByName.call(this, name);
      if (!hit && typeof CM.findModeByExtension === 'function' && name) {
        hit = CM.findModeByExtension(String(name).toLowerCase());
      }
      return hit;
    };
    CM.__tressoirModePatched = true;
  }

  function initFreeForm() {
    if (typeof global.CodeMirror === 'undefined') return;
    patchModeResolution();
    var ta = document.getElementById('ff-editor');
    if (!ta || ta.__cmMounted) return;
    ta.__cmMounted = true;
    // Namespace the feedback key by source basename (provider.ts emits data-source-name, e.g.
    // `PLAN`) so sibling artifacts sharing one folder's interactions.json don't collide. The SHIM
    // / preview harness sets no such attr -> falls back to the plain legacy key.
    var bodyData = (document.body && document.body.dataset) || {};
    var srcName = bodyData.sourceName || '';
    // New providers supply the complete, injective, bounded key. `sourceName` keeps the previous
    // namespaced runtime compatible; a plain fallback remains only for the standalone SHIM.
    var FF_KEY = bodyData.feedbackKey || ((srcName ? srcName + '-' : '') + 'free_form_feedback');
    var prior = readInteraction(FF_KEY);
    if (typeof prior === 'string') ta.value = prior;
    else if (prior && typeof prior.value === 'string') ta.value = prior.value;
    var cm = global.CodeMirror.fromTextArea(ta, {
      mode: { name: 'markdown', fencedCodeBlockHighlighting: true },
      lineWrapping: true, lineNumbers: false,
      placeholder: ta.getAttribute('placeholder') || ''
    });
    function resizeFeedbackEditor() {
      if (typeof cm.setSize !== 'function') return;
      var lines = typeof cm.lineCount === 'function' ? cm.lineCount() : cm.getValue().split('\n').length;
      var rows = Math.max(4, Math.min(8, lines));
      var rowHeight = typeof cm.defaultTextHeight === 'function' ? cm.defaultTextHeight() : 20;
      cm.setSize(null, Math.ceil(rows * rowHeight + 16));
    }
    resizeFeedbackEditor();
    var saveTimer = null;
    function flush() { if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; } noteInteraction(FF_KEY, cm.getValue()); }
    cm.on('change', function () { resizeFeedbackEditor(); if (saveTimer) clearTimeout(saveTimer); saveTimer = setTimeout(flush, 500); });
    cm.on('blur', flush);
    document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'hidden') flush(); });
    global.addEventListener('pagehide', flush);
    global.__ffcm = cm;
  }

  // Event delegation (morph-safe; bound to document ONCE).
  var _delegated = false;
  var _inputTimers = {};
  function localArtifactPath(value) {
    if (!value || value.charAt(0) === '#' || value.charAt(0) === '/' || value.slice(0, 2) === '//') return null;
    if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(value)) return null;
    var path = value.split('#')[0].split('?')[0];
    try { path = decodeURIComponent(path); } catch (_) { return null; }
    if (!path) return null;
    return path;
  }
  function prepareLocalArtifactLinks() {
    qsa('a[href]').forEach(function (anchor) {
      var path = anchor.getAttribute('data-tressoir-local-path') || localArtifactPath(anchor.getAttribute('href'));
      if (!path) return;
      // Remove the browser-navigation target entirely. The original contained path lives only in
      // data and is opened by the VS Code bridge below.
      anchor.setAttribute('data-tressoir-local-path', path);
      anchor.setAttribute('href', '#');
      anchor.removeAttribute('target');
    });
  }
  function copyCardMarkdown(index, button) {
    var source = _cardMarkdown[index];
    if (typeof source !== 'string') return;
    function report(ok) {
      if (!button) return;
      button.textContent = ok ? 'Copied' : 'Copy failed';
      if (button.__tressoirCopyReset) clearTimeout(button.__tressoirCopyReset);
      button.__tressoirCopyReset = setTimeout(function () { button.textContent = 'Copy Markdown'; }, 1600);
    }
    var clipboard = global.navigator && global.navigator.clipboard;
    if (clipboard && typeof clipboard.writeText === 'function') {
      try {
        Promise.resolve(clipboard.writeText(source)).then(function () { report(true); }, function () { report(false); });
        return;
      } catch (_) {}
    }
    try {
      var ta = document.createElement('textarea');
      ta.value = source;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0;';
      document.body.appendChild(ta);
      ta.select();
      var ok = typeof document.execCommand === 'function' && document.execCommand('copy');
      document.body.removeChild(ta);
      report(!!ok);
    } catch (_) { report(false); }
  }
  function initDelegation() {
    if (_delegated) return; _delegated = true;
    document.addEventListener('click', function (e) {
      var t = e.target; if (!t || !t.closest) return;
      var copyEl = t.closest('[data-copy-card]');
      if (copyEl) {
        e.preventDefault();
        e.stopImmediatePropagation();
        copyCardMarkdown(parseInt(copyEl.getAttribute('data-copy-card'), 10), copyEl);
        return;
      }
      var openFileEl = t.closest('[data-open-file]');
      if (openFileEl) {
        e.preventDefault();
        e.stopImmediatePropagation();
        var rel = openFileEl.getAttribute('data-open-file');
        if (rel && global.tressoirNotebook && typeof global.tressoirNotebook.openFile === 'function') global.tressoirNotebook.openFile(rel);
        return;
      }
      // Plain Markdown links and embedded images use the same contained open-file bridge as
      // explicit data-open-file controls. External URLs, fragments, absolute paths, and
      // traversal continue through normal browser handling (or are rejected by the provider).
      var localEl = t.closest('a[data-tressoir-local-path]') || t.closest('img[src]');
      if (localEl) {
        var localPath = localEl.tagName === 'A'
          ? localEl.getAttribute('data-tressoir-local-path')
          : localArtifactPath(localEl.getAttribute('src'));
        if (localPath) {
          e.preventDefault();
          e.stopImmediatePropagation();
          if (global.tressoirNotebook && typeof global.tressoirNotebook.openFile === 'function') {
            global.tressoirNotebook.openFile(localPath);
          }
          return;
        }
      }
      var ctxHead = t.closest('.ctx-item-head');
      if (ctxHead) {
        var item = ctxHead.closest('.ctx-item');
        if (item) { var on = item.classList.toggle('open'); ctxHead.setAttribute('aria-expanded', on ? 'true' : 'false'); }
        return;
      }
      var mHead = t.closest('.m-head');
      if (mHead) {
        var msx = mHead.closest('.milestone');
        if (msx) { var onm = msx.classList.toggle('selected'); mHead.setAttribute('aria-expanded', onm ? 'true' : 'false'); }
        return;
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var t = e.target; if (!t || !t.closest) return;
      if (t.closest('[data-copy-card]')) return;
      // ctx-item-head / openFile are real <button>s (native Enter/Space) — only the div-based
      // .m-head needs explicit keyboard activation.
      var mHead = t.closest('.m-head');
      if (mHead) { e.preventDefault(); var ms = mHead.closest('.milestone'); if (ms) { var onm = ms.classList.toggle('selected'); mHead.setAttribute('aria-expanded', onm ? 'true' : 'false'); } return; }
    });
    // :::input text box -> mark .resolved live + persist the raw string (debounced).
    document.addEventListener('input', function (e) {
      var t = e.target;
      if (!t || t.tagName !== 'TEXTAREA' || typeof t.getAttribute !== 'function') return;
      var key = t.getAttribute('data-input-key');
      if (!key) return;
      var dec = t.closest && t.closest('.m-decision');
      if (dec) dec.classList.toggle('resolved', !!String(t.value).trim());
      if (_inputTimers[key]) clearTimeout(_inputTimers[key]);
      var val = t.value;
      _inputTimers[key] = setTimeout(function () { noteInteraction(key, val); }, 500);
    });
    // Flush on blur (capture phase — blur does not bubble).
    document.addEventListener('blur', function (e) {
      var t = e.target;
      if (!t || t.tagName !== 'TEXTAREA' || typeof t.getAttribute !== 'function') return;
      var key = t.getAttribute('data-input-key');
      if (!key) return;
      if (_inputTimers[key]) { clearTimeout(_inputTimers[key]); _inputTimers[key] = null; }
      noteInteraction(key, t.value);
    }, true);
  }

  var _themeObserver = false;
  function applyBehaviors() {
    adaptTheme();
    if (!_themeObserver) {
      _themeObserver = true;
      try {
        var mo = new MutationObserver(adaptTheme);
        if (document.body) mo.observe(document.body, { attributes: true, attributeFilter: ['class'] });
      } catch (_) {}
    }
    initDelegation();
    applyRuntimeUI();
  }

  // Re-apply on every bridge render (full + morph): the projected DOM carries no init script.
  document.addEventListener('tressoir:render', function () { applyBehaviors(); });

  // ----------------------------------------------------------------- render (SHIM/standalone)
  function recreateScripts(container) {
    var scripts = Array.prototype.slice.call(container.querySelectorAll('script'))
      .filter(function (s) { return !s.hasAttribute('data-tressoir-script-ran'); });
    scripts.forEach(function (old) {
      var fresh = document.createElement('script');
      Array.prototype.slice.call(old.attributes).forEach(function (a) { fresh.setAttribute(a.name, a.value); });
      fresh.setAttribute('data-tressoir-script-ran', '1');
      fresh.textContent = old.textContent;
      old.replaceWith(fresh);
    });
  }
  function render(markdown, hostEl, bridge) {
    var host = hostEl || document.getElementById('app');
    if (bridge && !global.tressoirNotebook) { try { global.tressoirNotebook = bridge; } catch (_) {} }
    host.innerHTML = project(markdown);
    recreateScripts(host);
    applyBehaviors();
    try { document.dispatchEvent(new CustomEvent('tressoir:render', { detail: { phase: 'full' } })); } catch (_) {}
    return host;
  }

  // ----------------------------------------------------------------- lint
  // Pure static checker (NO DOM). Reuses the SAME remark + js-yaml parse as project() so its
  // verdict cannot diverge from what the editor actually renders. Returns an array of
  // { level: 'error'|'warn'|'info', line: <1-based>, msg: <string> }, flagging the silent
  // failure modes the renderer would otherwise swallow. Consumed by the Node
  // flows/scripts/check_md.js frontend so an agent can self-check a `.tressoir.md` before render.
  function lint(markdown) {
    var findings = [];
    function add(level, line, msg) { findings.push({ level: level, line: line || 0, msg: msg }); }
    function lineOf(n) { return (n && n.position && n.position.start && n.position.start.line) || 0; }

    // Flat text of an AST subtree (text + inline code), for length/heading checks.
    function plainText(n) {
      if (!n || typeof n !== 'object') return '';
      if (n.type === 'text' || n.type === 'inlineCode') return n.value || '';
      if (n.children) return n.children.map(plainText).join('');
      return '';
    }
    // The visible lifecycle label inside a card `state=` badge (HTML tags stripped).
    function stateLabel(a) {
      var s = (a && a.state != null) ? String(a.state) : '';
      return s.replace(/<[^>]*>/g, '').trim().toLowerCase();
    }
    // Lower-cased set of heading texts anywhere inside a card subtree.
    function headingTextsIn(node) {
      var found = {};
      (function rec(n) {
        if (!n || typeof n !== 'object') return;
        if (n.type === 'heading') found[plainText(n).trim().toLowerCase()] = true;
        if (n.children) n.children.forEach(rec);
      })(node);
      return found;
    }
    // Track explicit :::input keys to flag duplicates (warning, per accepted D2).
    var seenInputKeys = {};

    // Card lifecycle vs required sections (all warnings; exit status unchanged).
    function lintCard(node) {
      var line = lineOf(node), a = node.attributes || {};
      if (a.title != null && String(a.title).length > 200) {
        add('warn', line, '::::card title exceeds 200 characters — keep the visible title skimmable and put depth in the body');
      }
      if (a.title == null || String(a.title) === '') {
        add('warn', line, '::::card has no `title=` (the card head will be blank)');
      }
      var label = stateLabel(a);
      if (label === '') return;
      var headings = headingTextsIn(node);
      var hasPlanned = !!headings['planned changes'];
      var hasReport = !!headings['completion report'];
      if (label === 'planning' || label === 'implementing') {
        if (!hasPlanned) add('warn', line, '::::card is `' + label + '` but has no `#### Planned Changes` — show every planned edit as a named diff before handoff');
      } else if (label === 'review' || label === 'completed') {
        if (!hasReport && !hasPlanned) add('warn', line, '::::card is `' + label + '` but has no `#### Completion Report` or `#### Planned Changes`');
      } else if (label === 'tbd') {
        if (hasPlanned) add('warn', line, '::::card is `TBD` but already has `#### Planned Changes` — a TBD milestone should carry only a one-line overview');
      }
    }

    var extracted = extractMarkedCards(protectMath(markdown).markdown);
    extracted.findings.forEach(function (finding) { findings.push(finding); });
    var R = getRemark();
    var tree;
    try {
      tree = R.unified().use(R.remarkParse).use(R.remarkGfm)
        .use(R.remarkFrontmatter, ['yaml']).use(R.remarkDirective).parse(extracted.markdown);
    } catch (e) {
      add('error', 0, 'markdown failed to parse: ' + (e && e.message || e));
      return findings;
    }

    // ---- front-matter: optional control plane, never required presentation metadata.
    var fmNode = null;
    var fm = {};
    tree.children.forEach(function (n) { if (n.type === 'yaml' && !fmNode) fmNode = n; });
    if (fmNode) {
      var fmLine = lineOf(fmNode) || 1;
      var fmOk = true;
      try { fm = global.jsyaml.load(fmNode.value); }
      catch (e) { fmOk = false; add('error', fmLine, 'front-matter YAML is invalid: ' + (e && e.message || e)); }
      if (fmOk) {
        if (fm == null) {
          add('warn', fmLine, 'front-matter is empty');
        } else if (typeof fm !== 'object' || Array.isArray(fm)) {
          add('error', fmLine, 'front-matter is not a YAML mapping');
        } else {
          var ttype = fm.tressoir;
          var allowed = ['plan', 'research', 'interactive'];
          if (ttype != null && ttype !== '' && allowed.indexOf(String(ttype)) < 0) {
            add('warn', fmLine, 'front-matter `tressoir: ' + ttype + '` is not one of plan | research | interactive (type is optional)');
          }
          var links = Array.isArray(fm.links) ? fm.links : [];
          links.forEach(function (link) {
            if (/(?:tressoir-linear|katex|prism|codemirror)/i.test(String(link))) {
              add('error', fmLine, 'front-matter `links:` must not include standard linear, KaTeX, Prism, or CodeMirror assets; the extension loads them automatically');
            }
          });
        }
      }
    }

    var publicContent = tree.children.filter(function (n) { return n.type !== 'yaml'; });
    var leadingH1 = publicContent[0] && publicContent[0].type === 'heading' && publicContent[0].depth === 1;
    if (leadingH1) {
      if (!publicContent[1] || publicContent[1].type !== 'paragraph') {
        add('warn', lineOf(publicContent[0]), 'leading H1 has no immediately following lede paragraph');
      }
    } else if (!fm || !fm.title) {
      add('error', lineOf(publicContent[0]) || 1, 'artifact needs a leading H1; legacy front-matter `title` remains accepted only for existing files');
    }

    var seenControlKeys = {};
    var controlPattern = /data-tressoir-input\s*=\s*(["'])(.*?)\1/g;
    var controlMatch;
    while ((controlMatch = controlPattern.exec(markdown))) {
      var controlKey = controlMatch[2];
      var controlLine = markdown.slice(0, controlMatch.index).split(/\r?\n/).length;
      if (!controlKey) add('error', controlLine, 'data-tressoir-input needs a stable non-empty key');
      else if (seenControlKeys[controlKey]) add('warn', controlLine, 'data-tressoir-input key `' + controlKey + '` is reused; independent controls need independent keys');
      else seenControlKeys[controlKey] = true;
    }

    // ---- :::input — needs a `key=` (read-back) and a question (head text).
    function lintInput(node) {
      var line = lineOf(node), a = node.attributes || {};
      if (a.key == null || String(a.key) === '') {
        add('error', line, ':::input has no `key=` — the human answer cannot be read back from interactions.json');
      } else {
        var inputKey = String(a.key);
        if (inputKey === 'free_form_feedback' || /-free_form_feedback$/.test(inputKey)) {
          add('error', line, ':::input uses a reserved free-form feedback key — choose a different unique `key=`');
        } else if (seenInputKeys[inputKey]) {
          add('warn', line, ':::input key `' + inputKey + '` is used by more than one input — reuse binds both boxes to the same value; give each decision a unique `key=`');
        } else {
          seenInputKeys[inputKey] = true;
        }
      }
      var hasQ = (a.oneliner != null && a.oneliner !== '') || (a.question != null && a.question !== '');
      var qText = '';
      if (a.oneliner != null && a.oneliner !== '') qText = String(a.oneliner);
      else if (a.question != null && a.question !== '') qText = String(a.question);
      if (!hasQ) {
        var kids = node.children || [];
        var leadPara = kids.length && kids[0].type === 'paragraph';
        if (!leadPara) add('warn', line, ':::input has no question — add a leading paragraph or an `oneliner=` attribute (otherwise the head reads "Decision")');
        else qText = plainText(kids[0]);
      }
      if (qText.length > 200) {
        add('warn', line, ':::input question first paragraph exceeds 200 characters — keep the visible question short and put depth in the options/body');
      }
    }

    // ---- directive names + prose-in-plain-fence heuristic (one walk).
    var KNOWN = { card: 1, item: 1, input: 1 };
    var CODE_HINT = /[{};=]|=>|\bfunction\b|\bdef\b|\bclass\b|\breturn\b|\bimport\b|\bconst\b|\bvar\b|\blet\b|^\s*[-+*]\s|\/\//;
    function looksLikeProse(v) {
      var t = String(v == null ? '' : v).trim();
      if (t.length < 20) return false;
      if (CODE_HINT.test(t)) return false;
      var first = t.split(/\n/)[0].trim();
      if (first.split(/\s+/).length < 4) return false;
      return /[.?:]\s|[.?:]$/.test(first) || /\.\s/.test(t);
    }
    function walk(node, inItem) {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'code') {
        if (String(node.lang || '') === '' && looksLikeProse(node.value)) {
          add('warn', lineOf(node), 'looks like prose inside a plain ``` code fence — renders as a set-apart gray monospace box, not a paragraph (drop the fence, or move it to `description:`/lead text)');
        }
        if (!inItem) {
          // Use the same protected/grouped source that produced these AST line numbers. Display
          // math and marked-card extraction can otherwise shift later nodes away from their label.
          var sourceLines = String(extracted.markdown).split(/\r?\n/);
          var startLine = Math.max(0, lineOf(node) - 4);
          var nearby = sourceLines.slice(startLine, Math.max(0, lineOf(node) - 1)).join('\n');
          if (!/`[^`]*(?:\/|\.[A-Za-z0-9]+|\w+\([^)]*\))[^`]*`/.test(nearby)) {
            add('warn', lineOf(node), 'fenced code/diff has no nearby backticked file or symbol name');
          }
        }
      }
      var childInItem = inItem;
      if (node.type === 'containerDirective' || node.type === 'leafDirective' || node.type === 'textDirective') {
        var name = node.name || '';
        if (!KNOWN[name]) {
          add('warn', lineOf(node), 'unknown directive `:' + name + '` (known: card, item, input)');
        } else if (name === 'input') {
          lintInput(node);
        } else if (name === 'item') {
          var ia = node.attributes || {};
          if (ia.oneliner != null && String(ia.oneliner).length > 200) {
            add('warn', lineOf(node), ':::item oneliner exceeds 200 characters — keep the skimmable claim short and put depth inside the reveal body');
          }
          childInItem = true;
        } else if (name === 'card') {
          lintCard(node);
        }
      }
      if (node.children) node.children.forEach(function (c) { walk(c, childInItem); });
    }
    tree.children.forEach(function (n) { walk(n, false); });

    // ---- a directive marker that survived as PLAIN TEXT did not parse; the usual cause is
    // colon-nesting (a card that wraps items must use MORE colons than the items it wraps).
    var LEAK = /(^|\s):{2,}\s*(card|item|input)\b/;
    function scanLeak(node) {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'text' && LEAK.test(node.value || '')) {
        add('warn', lineOf(node), 'a directive marker survived as plain text (it did not parse) — usual cause: colon-nesting (a card that wraps items must use MORE colons, e.g. ::::card around :::item)');
      }
      if (node.children) node.children.forEach(scanLeak);
    }
    tree.children.forEach(scanLeak);

    function rk(l) { return l === 'error' ? 0 : l === 'warn' ? 1 : l === 'info' ? 2 : 9; }
    findings.sort(function (a, b) {
      if (a.line !== b.line) return a.line - b.line;
      return rk(a.level) - rk(b.level);
    });
    return findings;
  }

  global.TressoirMd = { project: project, render: render, applyBehaviors: applyBehaviors, lint: lint };
})(typeof window !== 'undefined' ? window : this);
