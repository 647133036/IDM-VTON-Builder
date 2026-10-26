import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { fmtAge, fmtAudience, fmtEthnicity, getLang, setLang, t } from './i18n.js'

beforeEach(() => {
  setLang('zh')
})

afterEach(() => {
  setLang('zh')
})

describe('t()', () => {
  it('returns Chinese copy by default', () => {
    expect(t('nav.bases')).toBe('模特底座')
  })

  it('interpolates named placeholders', () => {
    expect(t('btn.generate', { n: 3 })).toBe('一键生成 3 件')
  })

  it('falls back to the key when missing', () => {
    expect(t('no.such.key')).toBe('no.such.key')
  })
})

describe('setLang / getLang', () => {
  it('switches to English copy', () => {
    setLang('en')
    expect(getLang()).toBe('en')
    expect(t('nav.bases')).toBe('Mannequin Bases')
    expect(t('nav.storePrefix')).toBe('One garment, every mannequin')
  })

  it('unknown lang falls back to zh', () => {
    setLang('fr')
    expect(getLang()).toBe('zh')
  })
})

describe('fmtAudience / fmtAge', () => {
  it('maps audience keys in both languages', () => {
    expect(fmtAudience('women')).toBe('女装')
    setLang('en')
    expect(fmtAudience('women')).toBe('Women')
    expect(fmtAudience('men')).toBe('Men')
  })

  it('maps age-band keys', () => {
    expect(fmtAge('adult')).toBe('成人')
    setLang('en')
    expect(fmtAge('kids')).toBe('Kids')
  })
})

describe('fmtEthnicity', () => {
  const regions = [
    ['east_asia', '东亚', 'East Asian'],
    ['southeast_asia', '东南亚', 'Southeast Asian'],
    ['western_europe', '西欧', 'Western European'],
    ['eastern_europe', '东欧', 'Eastern European'],
    ['west_asia', '西亚', 'West Asian'],
    ['central_asia', '中亚', 'Central Asian'],
    ['africa', '非洲', 'African'],
    ['north_america', '北美', 'North American'],
    ['south_america', '南美', 'South American'],
    ['oceania', '澳洲', 'Oceanian'],
  ]

  it('translates all 10 preset regions in Chinese', () => {
    for (const [value, zh] of regions) {
      expect(fmtEthnicity(value)).toBe(zh)
    }
  })

  it('translates all 10 preset regions in English', () => {
    setLang('en')
    for (const [value, , en] of regions) {
      expect(fmtEthnicity(value)).toBe(en)
    }
  })

  it('keeps custom ethnicity labels as-is', () => {
    expect(fmtEthnicity('scandinavia', '北欧')).toBe('北欧')
    expect(fmtEthnicity('scandinavia')).toBe('scandinavia')
  })

  it('empty value returns empty string', () => {
    expect(fmtEthnicity('')).toBe('')
  })
})
