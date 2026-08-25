#!/usr/bin/env node
/*
 * Static checker for trusted `.tressoir.md` sources.
 *
 * It loads the same committed projector/lint runtime used by the VS Code
 * extension. No browser, DOM package, or npm install is required.
 *
 * Usage:
 *   node check_md.js <file.tressoir.md> [...] [--assets=<dir>] [--quiet]
 *
 * Exit codes:
 *   0 = no lint errors
 *   1 = at least one lint error
 *   2 = invocation, file-read, or runtime-load failure
 */
'use strict'

const fs = require('fs')
const path = require('path')
const vm = require('vm')

function defaultAssetsDir() {
  return path.resolve(__dirname, '..', 'references', 'runtime')
}

function loadRuntime(assetsDir) {
  const sandbox = {}
  sandbox.window = sandbox
  sandbox.self = sandbox
  sandbox.global = sandbox
  sandbox.console = console

  // lint() does not use the DOM, but the committed runtime performs two
  // harmless document operations at module load time.
  sandbox.document = {
    addEventListener: function () {},
    createElement: function () {
      let html = ''
      return {
        set innerHTML(value) { html = String(value == null ? '' : value) },
        get innerHTML() { return html },
        set textContent(value) { html = String(value == null ? '' : value) },
        get textContent() { return html },
      }
    },
  }

  vm.createContext(sandbox)
  const runtimeFiles = [
    path.join(assetsDir, 'core', 'tressoir-remark.umd.js'),
    path.join(assetsDir, 'core', 'js-yaml.min.js'),
    path.join(assetsDir, 'tressoir-md.js'),
  ]

  for (const file of runtimeFiles) {
    if (!fs.existsSync(file)) {
      throw new Error(`missing runtime asset: ${file}`)
    }
    vm.runInContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file })
  }

  if (!sandbox.TressoirMd || typeof sandbox.TressoirMd.lint !== 'function') {
    throw new Error('TressoirMd.lint was not exposed by the committed runtime')
  }
  return sandbox.TressoirMd
}

// Filesystem-level advisory check: when a projection `X.tressoir.md` has an
// exact sibling source `X.md` that is newer, the projection may be stale.
// Advisory only (warning) — never inspects or classifies interactions.json.
function sourceProjectionWarnings(target) {
  const findings = []
  if (!target.endsWith('.tressoir.md')) return findings
  const source = target.slice(0, -'.tressoir.md'.length) + '.md'
  let srcStat, projStat
  try { srcStat = fs.statSync(source) } catch (error) { return findings }
  try { projStat = fs.statSync(target) } catch (error) { return findings }
  if (srcStat.mtimeMs > projStat.mtimeMs) {
    findings.push({
      level: 'warn',
      line: 0,
      msg: `source \`${path.basename(source)}\` is newer than this projection — re-project and verify the change landed before handoff`,
    })
  }
  return findings
}

function usage() {
  console.error(
    'usage: node check_md.js <file.tressoir.md> [...] [--assets=<dir>] [--quiet]',
  )
}

function main(argv) {
  const args = argv.slice(2)
  const targets = []
  let assetsDir = defaultAssetsDir()
  let quiet = false

  for (const arg of args) {
    if (arg.startsWith('--assets=')) {
      assetsDir = path.resolve(arg.slice('--assets='.length))
    } else if (arg === '--quiet') {
      quiet = true
    } else if (arg === '-h' || arg === '--help') {
      usage()
      return 2
    } else if (arg.startsWith('-')) {
      console.error(`error: unknown option: ${arg}`)
      usage()
      return 2
    } else {
      targets.push(arg)
    }
  }

  if (targets.length === 0) {
    usage()
    return 2
  }

  let runtime
  try {
    runtime = loadRuntime(assetsDir)
  } catch (error) {
    console.error(`error: ${error && error.message ? error.message : error}`)
    return 2
  }

  let hadLintError = false
  let hadLoadFailure = false

  for (const target of targets) {
    let markdown
    try {
      markdown = fs.readFileSync(target, 'utf8')
    } catch (error) {
      console.error(
        `${target}\n  ERROR  could not read file: ${
          error && error.message ? error.message : error
        }\n`,
      )
      hadLoadFailure = true
      continue
    }

    let findings
    try {
      findings = runtime.lint(markdown) || []
    } catch (error) {
      console.error(
        `${target}\n  ERROR  lint crashed: ${
          error && error.stack ? error.stack : error
        }\n`,
      )
      hadLoadFailure = true
      continue
    }
    // Advisory filesystem warnings (exit status stays 0 for warnings).
    findings = findings.concat(sourceProjectionWarnings(target))

    const errors = findings.filter((finding) => finding.level === 'error').length
    const warnings = findings.filter((finding) => finding.level === 'warn').length
    const info = findings.length - errors - warnings
    if (errors > 0) hadLintError = true

    console.log(target)
    if (findings.length === 0) {
      console.log('  OK     no issues')
    } else {
      for (const finding of findings) {
        if (quiet && finding.level === 'info') continue
        console.log(
          `  ${String(finding.level).toUpperCase().padEnd(6)}line ${String(
            finding.line || '?',
          )}  ${finding.msg}`,
        )
      }
    }

    const summary = []
    if (errors) summary.push(`${errors} error${errors === 1 ? '' : 's'}`)
    if (warnings) summary.push(`${warnings} warning${warnings === 1 ? '' : 's'}`)
    if (info) summary.push(`${info} info`)
    console.log(`  -> ${findings.length ? summary.join(', ') : 'clean'}\n`)
  }

  if (hadLoadFailure) return 2
  return hadLintError ? 1 : 0
}

process.exit(main(process.argv))
