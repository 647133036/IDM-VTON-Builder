import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, it, expect } from 'vitest'

function extractFn(name, src) {
  const re = new RegExp('function ' + name + '\\(\\) \\{([\\s\\S]*?)\\n\\}')
  const m = src.match(re)
  if (!m) throw new Error(name + ' not found in main.js')
  return m[1]
}

function renderBasesView() {
  const src = readFileSync(resolve(process.cwd(), 'src/main.js'), 'utf8')
  const body = extractFn('basesView', src)
  const t = (k) => 'K'
  const escapeHtml = (s) => String(s ?? '')
  const fmtEthnicity = (v) => v
  const fmtAudience = (v) => v
  const fmtAge = (v) => v
  const ethnicityLabel = (v) => v
  const state = {
    baseBatchOpen: true,
    ethnicities: [{ value: 'east_asia', label: 'East Asia' }],
    bases: [
      {
        id: 'base_women_01',
        name: 'A',
        url: '/x.png',
        audience: 'women',
        age_band: 'adult',
        ethnicity: 'east_asia',
      },
    ],
    selectedBase: null,
    selectedBases: [],
  }
  return new Function(
    't',
    'escapeHtml',
    'fmtEthnicity',
    'fmtAudience',
    'fmtAge',
    'ethnicityLabel',
    'state',
    body
  )(t, escapeHtml, fmtEthnicity, fmtAudience, fmtAge, ethnicityLabel, state)
}

function countTags(html) {
  const count = (re) => (html.match(re) || []).length
  const tags = [
    'div',
    'label',
    'section',
    'button',
    'span',
    'select',
    'option',
    'input',
    'code',
    'em',
    'small',
    'a',
    'img',
  ]
  const selfClose = new Set(['input', 'img'])
  return tags.map((tag) => {
    const open = count(new RegExp('<' + tag + '(?=[\\s>])', 'g'))
    const close = selfClose.has(tag) ? open : count(new RegExp('</' + tag + '>', 'g'))
    return { tag, open, close, ok: selfClose.has(tag) || open === close }
  })
}

describe('basesView DOM regression', () => {
  it('renders balanced HTML (no misplaced ethnicity editor duplicate)', () => {
    const html = renderBasesView()
    const balance = countTags(html)
    const imbalanced = balance.filter((b) => !b.ok)
    expect(imbalanced).toEqual([])
  })

  it('keeps #base-batch-apply inside the batch box and after the batch row', () => {
    const html = renderBasesView()
    const box = html.indexOf('id="base-batch-box"')
    const apply = html.indexOf('id="base-batch-apply"')
    const bases = html.indexOf('class="bases"')
    expect(box).toBeGreaterThan(-1)
    expect(apply).toBeGreaterThan(box)
    expect(bases).toBeGreaterThan(apply)
  })

  it('removes the dead eth-add / eth-reset duplicate buttons', () => {
    const html = renderBasesView()
    expect(html.includes('id="eth-add"')).toBe(false)
    expect(html.includes('id="eth-reset"')).toBe(false)
  })
})
