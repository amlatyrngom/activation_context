(function () {
  'use strict'

  var boundControls = new Set()
  var pendingSaves = new Map()
  var feedbackEditors = new WeakMap()
  var feedbackDockOpen = false
  var feedbackDockRoutingInstalled = false
  var cardRoutingInstalled = false
  var decisionTrackingInstalled = false
  var autogrowTrackingInstalled = false
  var linkRoutingInstalled = false

  function bridge() {
    return window.tressoirNotebook || null
  }

  function context() {
    var api = bridge()
    if (api && typeof api.context === 'function') {
      try {
        return api.context()
      } catch (_) {}
    }
    return { feedbackKey: 'free_form_feedback', interactionsFile: 'interactions.json' }
  }

  function keyFor(control) {
    if (control.hasAttribute('data-tressoir-feedback')) return context().feedbackKey
    return control.getAttribute('data-tressoir-input') || ''
  }

  function valueOf(control) {
    if (control instanceof HTMLInputElement && control.type === 'checkbox') return control.checked
    return control.value
  }

  function applyValue(control, value) {
    if (control instanceof HTMLInputElement && control.type === 'checkbox') {
      control.checked = Boolean(value)
      return
    }
    if (typeof value === 'string' || typeof value === 'number') control.value = String(value)
  }

  function save(control) {
    var key = keyFor(control)
    if (!key) return
    var timer = pendingSaves.get(control)
    if (timer) window.clearTimeout(timer)
    pendingSaves.delete(control)

    var api = bridge()
    if (!api || typeof api.storeInteraction !== 'function') return
    try {
      api.storeInteraction(key, valueOf(control), context().interactionsFile)
    } catch (_) {}
  }

  function scheduleSave(control) {
    var key = keyFor(control)
    if (!key) return
    var previous = pendingSaves.get(control)
    if (previous) window.clearTimeout(previous)
    pendingSaves.set(control, window.setTimeout(function () { save(control) }, 300))
  }

  function bindControl(control) {
    if (boundControls.has(control)) return
    boundControls.add(control)

    var key = keyFor(control)
    if (!key) return
    var api = bridge()
    if (api && typeof api.getInteraction === 'function') {
      try {
        var saved = api.getInteraction(key, context().interactionsFile)
        if (saved !== undefined) applyValue(control, saved)
      } catch (_) {}
    }

    control.addEventListener('input', function () { scheduleSave(control) })
    control.addEventListener('change', function () { save(control) })
    control.addEventListener('blur', function () { save(control) })
  }

  function enhanceInteractions(root) {
    root.querySelectorAll('[data-tressoir-input], [data-tressoir-feedback]').forEach(bindControl)
  }

  function resizeAutogrow(textarea) {
    var range = (textarea.getAttribute('data-tressoir-autogrow') || '2:6').split(':')
    var minRows = Math.max(1, Number(range[0]) || 2)
    var maxRows = Math.max(minRows, Number(range[1]) || 6)
    textarea.rows = minRows

    var style = window.getComputedStyle(textarea)
    var fontSize = parseFloat(style.fontSize) || 16
    var lineHeight = parseFloat(style.lineHeight) || fontSize * 1.55
    var frame = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0) +
      (parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.borderBottomWidth) || 0)
    var minimum = lineHeight * minRows + frame
    var maximum = lineHeight * maxRows + frame

    textarea.style.height = 'auto'
    var needed = Math.max(textarea.scrollHeight, minimum)
    textarea.style.height = Math.min(needed, maximum) + 'px'
    textarea.style.overflowY = needed > maximum + 1 ? 'auto' : 'hidden'
  }

  function enhanceAutogrow(root) {
    root.querySelectorAll('textarea[data-tressoir-autogrow]').forEach(resizeAutogrow)
    if (autogrowTrackingInstalled) return
    autogrowTrackingInstalled = true
    document.addEventListener('input', function (event) {
      var textarea = event.target
      if (textarea instanceof HTMLTextAreaElement && textarea.hasAttribute('data-tressoir-autogrow')) {
        resizeAutogrow(textarea)
      }
    })
  }

  function updateDecision(decision) {
    var choices = Array.from(decision.querySelectorAll('input[type="checkbox"][data-tressoir-input]'))
    var resolved = choices.some(function (choice) { return choice.checked })
    var state = resolved ? 'resolved' : 'unresolved'
    decision.setAttribute('data-decision-state', state)
    var indicator = decision.querySelector('[data-decision-indicator]')
    if (indicator) indicator.textContent = resolved ? 'Resolved' : 'Unresolved'
  }

  function enhanceDecisions(root) {
    root.querySelectorAll('[data-tressoir-decision]').forEach(updateDecision)
    if (decisionTrackingInstalled) return
    decisionTrackingInstalled = true
    document.addEventListener('input', function (event) {
      var target = event.target
      if (!(target instanceof HTMLInputElement) || target.type !== 'checkbox') return
      var decision = target.closest('[data-tressoir-decision]')
      if (decision) updateDecision(decision)
    })
  }

  function enhanceFeedbackEditors(root) {
    if (!window.CodeMirror || typeof window.CodeMirror.fromTextArea !== 'function') return
    root.querySelectorAll('textarea[data-tressoir-feedback]').forEach(function (textarea) {
      var existing = feedbackEditors.get(textarea)
      if (existing && existing.getWrapperElement().isConnected) return
      if (existing) {
        try { existing.toTextArea() } catch (_) { textarea.style.display = '' }
      }

      var editor = window.CodeMirror.fromTextArea(textarea, {
        mode: { name: 'markdown', highlightFormatting: true },
        lineWrapping: true,
        viewportMargin: 20
      })
      editor.getInputField().setAttribute('aria-label', 'Feedback Form (Markdown)')
      editor.on('change', function () {
        editor.save()
        textarea.dispatchEvent(new Event('input', { bubbles: true }))
      })
      editor.on('blur', function () {
        editor.save()
        save(textarea)
      })
      feedbackEditors.set(textarea, editor)
    })
  }

  function renderFeedbackDock(focusEditor) {
    var dock = document.querySelector('[data-feedback-dock]')
    if (!dock) return
    var panel = dock.querySelector('[data-feedback-panel]')
    var toggle = dock.querySelector('[data-feedback-toggle]')
    if (!panel || !toggle) return

    panel.hidden = !feedbackDockOpen
    toggle.setAttribute('aria-expanded', feedbackDockOpen ? 'true' : 'false')
    toggle.setAttribute('aria-label', feedbackDockOpen ? 'Close feedback form' : 'Open feedback form')

    if (feedbackDockOpen && focusEditor) {
      var textarea = panel.querySelector('textarea[data-tressoir-feedback]')
      var editor = textarea && feedbackEditors.get(textarea)
      if (editor) {
        var scheduleFrame = window.requestAnimationFrame || function (callback) { window.setTimeout(callback, 0) }
        scheduleFrame(function () {
          editor.refresh()
          editor.focus()
        })
      }
    }
  }

  function installFeedbackDockRouting() {
    if (feedbackDockRoutingInstalled) return
    feedbackDockRoutingInstalled = true

    document.addEventListener('click', function (event) {
      var target = event.target instanceof Element ? event.target : null
      var toggle = target && target.closest('[data-feedback-toggle]')
      var close = target && target.closest('[data-feedback-close]')
      if (!toggle && !close) return
      event.preventDefault()

      if (toggle) {
        feedbackDockOpen = !feedbackDockOpen
        renderFeedbackDock(feedbackDockOpen)
        return
      }

      feedbackDockOpen = false
      renderFeedbackDock(false)
      var trigger = document.querySelector('[data-feedback-toggle]')
      if (trigger) trigger.focus()
    })

    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape' || !feedbackDockOpen) return
      event.preventDefault()
      feedbackDockOpen = false
      renderFeedbackDock(false)
      var trigger = document.querySelector('[data-feedback-toggle]')
      if (trigger) trigger.focus()
    })
  }

  function installCardRouting() {
    if (cardRoutingInstalled) return
    cardRoutingInstalled = true

    document.addEventListener('click', function (event) {
      var target = event.target instanceof Element ? event.target : null
      var gutter = target && target.closest('[data-card-collapse]')
      if (!gutter) return
      var card = gutter.closest('details.card')
      if (!card || !card.open) return
      event.preventDefault()
      event.stopPropagation()
      card.open = false
      var summary = card.querySelector(':scope > summary')
      if (summary) {
        try {
          summary.focus({ preventScroll: true })
        } catch (_) {
          summary.focus()
        }
      }
    })
  }

  function ensureScrollRegions(root) {
    root.querySelectorAll('.tressoir-document table, .tressoir-document svg').forEach(function (element) {
      if (element.hasAttribute('data-tressoir-inline')) return
      if (element.closest('.scroll-region, .katex, .CodeMirror')) return
      var wrapper = document.createElement('div')
      wrapper.className = 'scroll-region'
      wrapper.setAttribute('data-tressoir-auto-scroll', '')
      wrapper.setAttribute('tabindex', '0')
      wrapper.setAttribute('role', 'region')
      var caption = element.querySelector && element.querySelector('caption')
      var label = element.getAttribute('aria-label') || (caption && caption.textContent) ||
        (element.tagName === 'TABLE' ? 'Scrollable table' : 'Scrollable diagram')
      wrapper.setAttribute('aria-label', label.trim())
      element.parentNode.insertBefore(wrapper, element)
      wrapper.appendChild(element)
    })
  }

  function enhanceMath(root) {
    if (typeof window.renderMathInElement !== 'function') return
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
        ignoredClasses: ['katex'],
        trust: false,
        strict: 'warn',
        throwOnError: false
      })
    } catch (_) {}
  }

  function enhanceCode(root) {
    if (!window.Prism || typeof window.Prism.highlightAllUnder !== 'function') return
    try { window.Prism.highlightAllUnder(root) } catch (_) {}
  }

  function neutralizeLocalLinks(root) {
    var api = bridge()
    if (!api || typeof api.openFile !== 'function') return
    root.querySelectorAll('a[data-tressoir-open-file]').forEach(function (link) {
      var path = link.getAttribute('data-tressoir-local-path') || link.getAttribute('href')
      if (!path || path === '#') return
      link.setAttribute('data-tressoir-local-path', path)
      link.setAttribute('href', '#')
      link.removeAttribute('target')
    })
  }

  function installLinkRouting() {
    if (linkRoutingInstalled) return
    linkRoutingInstalled = true
    document.addEventListener('click', function (event) {
      var target = event.target instanceof Element ? event.target.closest('[data-tressoir-open-file]') : null
      if (!(target instanceof HTMLAnchorElement)) return
      var api = bridge()
      if (!api || typeof api.openFile !== 'function') return
      var path = target.getAttribute('data-tressoir-local-path')
      if (!path) return
      event.preventDefault()
      event.stopImmediatePropagation()
      api.openFile(path)
    })
  }

  function enhance(root) {
    ensureScrollRegions(root)
    enhanceMath(root)
    enhanceCode(root)
    enhanceInteractions(root)
    enhanceAutogrow(root)
    enhanceDecisions(root)
    enhanceFeedbackEditors(root)
    renderFeedbackDock(false)
    installCardRouting()
    installFeedbackDockRouting()
    neutralizeLocalLinks(root)
    installLinkRouting()
  }

  document.addEventListener('tressoir:render', function () { enhance(document) })
  window.addEventListener('pagehide', function () {
    Array.from(pendingSaves.keys()).forEach(save)
  })

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { enhance(document) }, { once: true })
  } else {
    enhance(document)
  }
})()
