import './style.css'
import { t, getLang, setLang, fmtAudience, fmtAge, fmtEthnicity } from './i18n'

const TOKEN_KEY = 'atelier_token'
const AUDIENCE_LABEL = {
  auto: 'auto',
  women: 'women',
  men: 'men',
  girls: 'girls',
  boys: 'boys',
  baby: 'baby',
}
const AGE_LABEL = { adult: 'adult', kids: 'kids', baby: 'baby' }
function ethnicityLabel(value) {
  const hit = (state.ethnicities || []).find((e) => e.value === value)
  return hit ? fmtEthnicity(hit.value, hit.label) : (value || '')
}
const state = {
  token: localStorage.getItem(TOKEN_KEY) || '',
  view: 'batch',
  user: null,
  bases: [],
  jobs: [],
  stats: {},
  settings: {},
  stores: [],
  ethnicities: [],
  selectedBase: '',
  selectedBases: [],
  baseBatchOpen: false,
  ethnicity: '',
  storeCode: '',
  files: [],
  audience: 'auto',
  garmentType: 'auto',
  outputFormat: 'jpg',
  background: 'white',
  error: '',
  previewJob: '',
  pushJob: '',
  license: null,
  uploads: [],
  uploadCategories: {},
  selectedUploads: [],
}

async function api(path, options = {}) {
  const headers = options.headers || {}
  if (state.token) headers.Authorization = `Bearer ${state.token}`
  const res = await fetch(path, { ...options, headers })
  if (res.status === 401) {
    logout()
    throw new Error(t('login.need'))
  }
  if (!res.ok) {
    let detail = t('login.failed')
    try {
      const data = await res.json()
      detail = data.detail || detail
    } catch {}
    throw new Error(detail)
  }
  const type = res.headers.get('content-type') || ''
  if (type.includes('application/json')) return res.json()
  return res
}

function logout() {
  state.token = ''
  state.user = null
  localStorage.removeItem(TOKEN_KEY)
  render()
}

function appRoot() {
  return document.querySelector('#app')
}

function render() {
  if (!state.token) {
    appRoot().innerHTML = loginView()
    bindLogin()
    return
  }
  appRoot().innerHTML = dashboardView()
  bindDashboard()
}

function loginView() {
  return `
    <div class="login">
      <div class="login-card">
        <h1>MannequinForge</h1>
        <p>${t('login.subtitle')}</p>
        <label>${t('login.acc')}<input id="u" value="admin" /></label>
        <label>${t('login.pwd')}<input id="p" type="password" value="admin123" /></label>
        <button class="btn gold" id="login">${t('login.btn')}</button>
        <button class="btn ghost" id="login-lang" style="width:100%;margin-top:10px">${getLang() === 'zh' ? 'Switch to English' : '切换到中文'}</button>
        <p class="err" id="login-err"></p>
      </div>
    </div>
  `
}

function dashboardView() {
  return `
    <div class="wrap">
      <header class="top">
        <div class="brand">
          <h1>MannequinForge</h1>
          <span>${state.user?.name || 'Store Admin'} · ${t('nav.storePrefix')}${state.storeCode ? ` · ${t('nav.embedStore')} ${escapeHtml(state.storeCode)}` : ''}</span>
        </div>
        <nav class="nav">
          <button data-view="batch" class="${state.view === 'batch' ? 'on' : ''}">${t('nav.batch')}</button>
          <button data-view="bases" class="${state.view === 'bases' ? 'on' : ''}">${t('nav.bases')}</button>
          <button data-view="queue" class="${state.view === 'queue' ? 'on' : ''}">${t('nav.queue')}</button>
          <button data-view="embed" class="${state.view === 'embed' ? 'on' : ''}">${t('nav.embed')}</button>
          <button id="lang-switch" class="${''}">${getLang() === 'zh' ? 'EN' : '中'}</button>
          <button id="logout" class="${''}">${t('nav.logout')}</button>
        </nav>
      </header>
      ${licenseBanner()}
      <section class="kpis">
        ${kpi(t('kpi.queued'), state.stats.queued)}
        ${kpi(t('kpi.running'), state.stats.running)}
        ${kpi(t('kpi.done'), state.stats.done)}
        ${kpi(t('kpi.failed'), state.stats.failed)}
        ${kpi(t('kpi.bases'), state.stats.bases)}
      </section>
      ${state.view === 'batch' ? batchView() : ''}
      ${state.view === 'bases' ? basesView() : ''}
      ${state.view === 'queue' ? queueView() : ''}
      ${state.view === 'embed' ? embedView() : ''}
      ${state.previewJob ? previewModal() : ''}
      ${state.pushJob ? pushModal() : ''}
    </div>
  `
}

function activeEngineLabel(l) {
  if (!l.gpu_available) return t('engine.noGpu')
  if (l.sdxl_available) return t('engine.sdxlOk')
  return t('engine.sdxlNo')
}

function licenseBanner() {
  if (!state.license) return ''
  const l = state.license
  const engine = activeEngineLabel(l)
  const link = '<a href="/docs/LICENSE_COMPLIANCE.md" target="_blank">'
  const linkEnd = '</a>'
  if (l.license_mode === 'commercial') {
    return `<div class="banner warn"><strong>${t('banner.commercial')}</strong>：${t('banner.commercialText', { engine, link, linkEnd })}</div>`
  }
  if (l.license_mode === 'research') {
    const idmNote = t('banner.idmNote')
    const sdxlNote = t('banner.sdxlNote')
    if (l.idm_vton_available) {
      return `<div class="banner warn"><strong>${t('banner.research')}</strong>：${idmNote}；${sdxlNote}。${t('banner.researchIdmOn', { engine })}</div>`
    }
    return `<div class="banner warn"><strong>${t('banner.research')}</strong>：${idmNote}；${sdxlNote}。${t('banner.researchIdmOff', { engine })}</div>`
  }
  return `<div class="banner warn"><strong>${t('banner.unset')}</strong>：${t('banner.unsetText', { link, linkEnd })}</div>`
}

function previewModal() {
  const job = state.jobs.find((j) => j.id === state.previewJob)
  if (!job) return ''
  const base = state.bases.find((b) => b.id === job.base_id)
  return `
    <div class="modal" id="preview-modal">
      <div class="modal-inner">
        <div class="modal-head">
          <div>
            <h3>${t('pv.title')}</h3>
            <small>${escapeHtml(job.title || job.source_name)} · ${fmtAudience(job.audience) || job.audience} · ${escapeHtml(job.garment_type)}</small>
          </div>
          <button class="btn ghost small" id="close-preview">${t('pv.close')}</button>
        </div>
        <div class="modal-body">
          <div class="preview-col">
            <span class="preview-label">${t('pv.source')}</span>
            <img class="preview-img" src="${job.source_url}" alt="${t('pv.source')}" />
          </div>
          <div class="preview-col main">
            <span class="preview-label">${t('pv.result')} · ${base ? escapeHtml(base.name) : t('pv.base')}</span>
            <img class="preview-img big" src="${job.result_url}" alt="${t('pv.result')}" />
          </div>
        </div>
        <div class="modal-foot">
          <a class="btn gold" href="/api/download/${job.id}" download>${t('pv.download')} ${job.result_url.split('.').pop().toUpperCase()}</a>
          <button class="btn ghost" id="close-preview-2">${t('pv.close')}</button>
        </div>
      </div>
    </div>
  `
}

function pushModal() {
  const job = state.jobs.find((j) => j.id === state.pushJob)
  if (!job) return ''
  const stores = state.stores || []
  const doneStoreIds = job.pushed_stores || []
  return `
    <div class="modal" id="push-modal">
      <div class="modal-inner">
        <div class="modal-head">
          <div>
            <h3>${t('push.title')}</h3>
            <small>${escapeHtml(job.title || job.source_name)}</small>
          </div>
          <button class="btn ghost small" id="close-push">${t('push.close')}</button>
        </div>
        <div class="modal-body">
          ${stores.length ? stores.map((s) => {
            const checked = doneStoreIds.includes(s.id) ? 'checked' : ''
            return `<label class="store-opt"><input type="checkbox" data-store="${s.id}" ${checked} /> <b>${escapeHtml(s.name)}</b> <span class="hint">${escapeHtml(s.platform || 'other')}${s.product_id ? ' · #' + escapeHtml(s.product_id) : ''}</span></label>`
          }).join('') : `<p class="hint">${t('push.nostore')}</p>`}
          <label class="store-opt"><span>${t('push.productId')}</span><input id="push-product-id" placeholder="${t('push.productIdPh')}" /></label>
        </div>
        <div class="modal-foot">
          <button class="btn gold" id="confirm-push" ${stores.length ? '' : 'disabled'}>${t('push.confirm')}</button>
          <button class="btn ghost" id="close-push-2">${t('push.close')}</button>
        </div>
      </div>
    </div>
  `
}

function kpi(label, value) {
  return `<div class="kpi"><b>${value ?? 0}</b><span>${label}</span></div>`
}

function batchView() {
  return `
    <div class="grid">
      <aside class="panel">
        <h3>${t('step.base')}</h3>
        <div class="row">
          <label>${t('eth.filter')}
            <select id="eth-filter">
              <option value="">${t('eth.all')}</option>
              ${(state.ethnicities || []).map((e) => `<option value="${e.value}" ${state.ethnicity === e.value ? 'selected' : ''}>${escapeHtml(fmtEthnicity(e.value, e.label))}</option>`).join('')}
            </select>
          </label>
        </div>
        <div class="bases" id="base-list">${baseCards()}</div>
        <h3 style="margin-top:16px">${t('step.output')}</h3>
        <div class="row">
          <label>${t('form.audience')}
            <select id="audience">
              ${Object.keys(AUDIENCE_LABEL).map((k) => `<option value="${k}" ${state.audience === k ? 'selected' : ''}>${fmtAudience(k)}</option>`).join('')}
            </select>
          </label>
          <label>${t('form.garment')}
            <select id="gtype">
              <option value="auto">${t('opt.auto')}</option>
              <option value="upper">${t('opt.upper')}</option>
              <option value="dress">${t('opt.dress')}</option>
              <option value="lower">${t('opt.lower')}</option>
              <option value="onesie">${t('opt.onesie')}</option>
            </select>
          </label>
        </div>
        <div class="row">
          <label>${t('form.fmt')}
            <select id="fmt">
              <option value="jpg">${t('opt.jpg')}</option>
              <option value="png">${t('opt.png')}</option>
            </select>
          </label>
          <label>${t('form.bg')}
            <select id="bg">
              <option value="white">${t('opt.white')}</option>
              <option value="transparent">${t('opt.transparent')}</option>
            </select>
          </label>
        </div>
        <p class="hint">${t('hint.output')}</p>
      </aside>
      <section class="panel">
        <h3>${t('step.drop')}</h3>
        <div class="drop" id="drop">${t('drop.hint')}<br><small>${t('drop.formats')}</small></div>
        <input id="file" type="file" accept="image/*" multiple hidden />
        <div class="thumbs" id="file-thumbs"></div>
        <div class="row" style="margin-top:12px">
          <button class="btn gold" id="run">${t('btn.generate', { n: state.files.length })}</button>
          <button class="btn ghost" id="samples">${t('btn.samples')}</button>
          <button class="btn ghost" id="clear">${t('btn.clear')}</button>
        </div>
        ${state.selectedUploads.length ? `
        <div class="row" style="margin-top:12px;padding:10px;background:rgba(212,180,131,.1);border-radius:10px">
          <small>${t('sel.count', { n: state.selectedUploads.length })}</small>
          <button class="btn gold small" id="add-selected">${t('btn.addSel')}</button>
        </div>` : ''}
        <p class="err" id="batch-err">${state.error}</p>
        <h3 style="margin-top:18px">${t('lib.upload')}</h3>
        <div class="upload-lib" id="upload-lib">${uploadLibrary()}</div>
        <h3 style="margin-top:18px">${t('lib.cloth')}</h3>
        <div class="cloth-lib" id="cloth-lib">${clothLibrary()}</div>
        <h3 style="margin-top:18px">${t('lib.recent')}</h3>
        <div class="queue">${jobCards(state.jobs.slice(0, 8))}</div>
      </section>
    </div>
  `
}

function basesView() {
  return `
    <div class="grid">
      <section class="panel">
        <h3>${t('base.title')}</h3>
        <p class="hint">${t('base.hint')}</p>
        <div class="row" style="margin-bottom:8px">
          <button class="btn ghost small" id="base-batch-sel">${t('base.batchBtn')}</button>
          <span id="base-batch-count" style="font-size:12px;opacity:.7"></span>
        </div>
        <div id="base-batch-box" class="eth-manage" style="display:${state.baseBatchOpen ? 'block' : 'none'}">
          <div class="row">
            <span class="hint" id="base-batch-tip">${t('base.batchTip')}</span>
          </div>
          <div class="row">
            <label>${t('form.audience')}
              <select id="bbaud">
                <option value="">${t('base.keep')}</option>
                <option value="women">${t('aud.women')}</option>
                <option value="men">${t('aud.men')}</option>
                <option value="girls">${t('aud.girls')}</option>
                <option value="boys">${t('aud.boys')}</option>
                <option value="baby">${t('aud.baby')}</option>
              </select>
            </label>
            <label>${t('form.ageband')}
              <select id="bbage">
                <option value="">${t('base.keep')}</option>
                <option value="adult">${t('age.adult')}</option>
                <option value="kids">${t('age.kids')}</option>
                <option value="baby">${t('age.baby')}</option>
              </select>
            </label>
            <label>${t('form.ethnicity')}
              <select id="bbeth">
                <option value="">${t('base.keep')}</option>
                ${(state.ethnicities || []).map((e) => `<option value="${e.value}">${escapeHtml(fmtEthnicity(e.value, e.label))}</option>`).join('')}
              </select>
            </label>
          </div>
          <button class="btn gold small" id="base-batch-apply">${t('base.apply')}</button>
        </div>
        <div class="bases">${state.bases.map((b) => `
          <div class="base ${state.selectedBase === b.id ? 'sel' : ''} ${state.selectedBases.includes(b.id) ? 'batch-sel' : ''}" data-id="${b.id}">
            <img src="${b.url}" alt="${escapeHtml(b.name)}" />
            <em>${escapeHtml(b.name)}</em>
            <small>${fmtAudience(b.audience) || b.audience || '—'} · ${fmtAge(b.age_band) || b.age_band || '—'} · ${escapeHtml((() => { const hit = state.ethnicities.find(x => x.value === b.ethnicity); return hit ? fmtEthnicity(hit.value, hit.label) : (b.ethnicity || '—') })())}</small>
            <div class="row" style="flex:none">
              <label title="${t('base.batchTitle')}"><input type="checkbox" data-bsel="${b.id}" ${state.selectedBases.includes(b.id) ? 'checked' : ''} /> ${t('base.batch')}</label>
              <button class="btn ghost small" data-del="${b.id}" title="${t('base.retireTitle')}">${t('base.retire')}</button>
            </div>
          </div>`).join('')}</div>
      </section>
      <section class="panel">
        <h3>${t('base.upload')}</h3>
        <label>${t('form.name')}<input id="bname" placeholder="${t('base.namePh')}" /></label>
        <div class="row">
          <label>${t('form.audience')}
            <select id="baud">
              <option value="women">${t('aud.women')}</option>
              <option value="men">${t('aud.men')}</option>
              <option value="girls">${t('aud.girls')}</option>
              <option value="boys">${t('aud.boys')}</option>
              <option value="baby">${t('aud.baby')}</option>
            </select>
          </label>
          <label>${t('form.ageband')}
            <select id="bage">
              <option value="adult">${t('age.adult')}</option>
              <option value="kids">${t('age.kids')}</option>
              <option value="baby">${t('age.baby')}</option>
            </select>
          </label>
          <label>${t('form.ethnicity')}
            <select id="beth">
              ${(state.ethnicities || []).map((e) => `<option value="${e.value}">${escapeHtml(fmtEthnicity(e.value, e.label))}</option>`).join('')}
            </select>
          </label>
        </div>
        <button class="btn ghost small" id="eth-manage" type="button">${t('base.ethManage')}</button>
        <div id="eth-manage-box" class="eth-manage" style="display:none">
          <div class="row">
            <input id="new-eth-value" placeholder="${t('eth.newValue')}" />
            <input id="new-eth-label" placeholder="${t('eth.newLabel')}" />
            <button class="btn gold small" id="add-eth">${t('eth.add')}</button>
          </div>
          <div class="eth-list">
            ${(state.ethnicities || []).map((e) => `
              <div class="eth-item">
                <span>${escapeHtml(fmtEthnicity(e.value, e.label))}</span>
                <code>${escapeHtml(e.value)}</code>
                <div class="row" style="flex:none">
                  <button class="btn ghost small eth-rename" data-value="${e.value}" data-label="${escapeHtml(e.label)}">${t('eth.rename')}</button>
                  <button class="btn ghost small eth-del" data-value="${e.value}">${t('eth.del')}</button>
                </div>
              </div>`).join('')}
          </div>
        </div>
        <label>${t('form.notes')}<input id="bnotes" placeholder="${t('base.notesPh')}" /></label>
        <label>${t('form.file')}<input id="bfile" type="file" accept="image/*" /></label>
        <button class="btn gold" id="add-base">${t('base.save')}</button>
        <p class="err" id="base-err"></p>
      </section>
    </div>
  `
}

function queueView() {
  const done = state.jobs.filter((j) => j.status === 'done')
  return `
    <section class="panel">
      <h3>${t('queue.title')}</h3>
      <div class="row" style="margin-bottom:12px">
        <button class="btn ghost small" id="dl-all">${t('queue.dlAll', { n: done.length })}</button>
      </div>
      <div class="queue">${jobCards(state.jobs)}</div>
    </section>
  `
}

function embedView() {
  const stores = state.stores || []
  const listHtml = stores.length ? stores.map((s) => {
    const snippet = `<iframe src="${location.origin}/embed.html?store=${s.id}" style="width:100%;min-height:720px;border:0;border-radius:16px"></iframe>`
    return `
      <div class="store-card">
        <div class="store-head">
          <b>${escapeHtml(s.name)}</b>
          <span class="tag">${escapeHtml(s.platform || 'other')}</span>
          <span class="hint">${t('embed.product')}${s.product_id ? '#' + escapeHtml(s.product_id) : t('embed.bySku')}</span>
          <button class="btn ghost small" data-del-store="${s.id}">${t('embed.delStore')}</button>
        </div>
        <p class="hint">${t('embed.storeCode')}</p>
        <div class="code">${escapeHtml(snippet)}</div>
        <p class="hint">${t('embed.base', { url: escapeHtml('/embed.html?store=' + s.id) })}</p>
      </div>
    `
  }).join('') : `<p class="hint">${t('embed.empty')}</p>`
  return `
    <div class="grid">
      <section class="panel">
        <h3>${t('embed.title')}</h3>
        <p class="hint">${t('embed.hint')}</p>
        ${listHtml}
        <h3 style="margin-top:18px">${t('embed.add')}</h3>
        <label>${t('embed.sname')}<input id="sname" placeholder="${t('embed.snamePh')}" /></label>
        <label>${t('embed.platform')}
            <select id="splatform">
              <option value="shopify">Shopify</option>
              <option value="wordpress">WordPress / WooCommerce</option>
              <option value="tiktok">TikTok Shop</option>
              <option value="taobao">${t('embed.taobao')}</option>
              <option value="other">${t('embed.other')}</option>
            </select>
          </label>
        <label>${t('embed.shopUrl')}<input id="sshop" placeholder="${t('embed.shopUrlPh', { shop: '{shop}' })}" /></label>
        <label>${t('embed.api')}<input id="sapi" placeholder="${t('embed.apiPh')}" /></label>
        <label>${t('embed.productId')}<input id="sprod" placeholder="${t('embed.productIdPh')}" /></label>
        <button class="btn gold" id="add-store">${t('embed.add')}</button>
        <p class="err" id="store-err"></p>
      </section>
      <section class="panel">
        <h3>${t('embed.pushTitle')}</h3>
        <p class="hint">${t('embed.pushHint')}</p>
        <div class="code">POST /api/auth/login
GET  /api/bases
POST /api/jobs/batch
GET  /api/jobs
POST /api/jobs/{id}/push   body: {"store_ids":["x"],"product_id":"123"}
GET  /api/download/{id}
GET/POST/PATCH/DELETE /api/stores</div>
        <p class="hint">${t('embed.pushAdapter')}</p>
        <p class="hint">${t('embed.prodNote')}</p>
      </section>
    </div>
  `
}

function baseCards() {
  const list = state.ethnicity ? state.bases.filter((b) => b.ethnicity === state.ethnicity) : state.bases
  return list.map((b) => `
    <div class="base ${state.selectedBase === b.id ? 'sel' : ''}" data-id="${b.id}">
      <img src="${b.url}" alt="${escapeHtml(b.name)}" />
      <em>${ethnicityLabel(b.ethnicity) ? ethnicityLabel(b.ethnicity) + ' · ' : ''}${fmtAudience(b.audience) || escapeHtml(b.name)}</em>
    </div>
  `).join('')
}

const CLOTH_NAMES = ['04469_00.jpg','04743_00.jpg','09133_00.jpg','09163_00.jpg','09164_00.jpg','09166_00.jpg','09176_00.jpg','09236_00.jpg','09256_00.jpg','09263_00.jpg','09266_00.jpg','09290_00.jpg','09305_00.jpg','10165_00.jpg','14627_00.jpg','14673_00.jpg']

function clothLibrary() {
  return CLOTH_NAMES.map((n) => `
    <div class="cloth-item" data-cloth="${n}">
      <img src="/examples/cloth/${n}" alt="${n}" loading="lazy" />
      <span>${n.replace(/\.jpg$/,'')}</span>
    </div>
  `).join('')
}

function uploadLibrary() {
  const cats = state.uploadCategories ? Object.keys(state.uploadCategories).filter(c => c !== 'default') : []
  return `
    <div class="upload-toolbar">
      <button class="btn ghost small" id="upload-lib-add">${t('upl.add')}</button>
      <input type="file" id="upload-lib-file" accept="image/*" multiple hidden />
      <select id="cat-filter" class="cat-filter">
        <option value="">${t('upl.all')}</option>
        <option value="default">${t('upl.unsorted')}</option>
        ${cats.map(c => `<option value="${c}">${c}</option>`).join('')}
      </select>
      <button class="btn ghost small" id="cat-move">${t('upl.move')}</button>
      <select id="cat-move-select" style="display:none">
        <option value="">${t('upl.choose')}</option>
        ${cats.map(c => `<option value="${c}">${c}</option>`).join('')}
        <option value="__new__">${t('upl.new')}</option>
      </select>
      <input type="text" id="cat-name" placeholder="${t('upl.newName')}" style="width:120px;display:none" />
      <button class="btn ghost small" id="cat-manage">${t('upl.manage')}</button>
    </div>
    <div id="cat-manage-box" class="cat-manage" style="display:none">
      <div class="row">
        <input id="new-cat-name" placeholder="${t('upl.newName')}" />
        <button class="btn gold small" id="add-cat">${t('upl.create')}</button>
      </div>
      <div class="cat-list">
        ${cats.length ? cats.map(c => `
          <div class="cat-item">
            <code>${escapeHtml(c)}</code> <span class="hint">${t('upl.count', { n: state.uploadCategories[c] || 0 })}</span>
            <div class="row" style="flex:none">
              <button class="btn ghost small cat-rename" data-cat="${escapeHtml(c)}">${t('eth.rename')}</button>
              <button class="btn ghost small cat-del" data-cat="${escapeHtml(c)}">${t('eth.del')}</button>
            </div>
          </div>`).join('') : `<p class="hint">${t('upl.empty')}</p>`}
      </div>
    </div>
    <div class="upload-list">
      ${!state.uploads.length ? `<p class="hint">${t('upl.noUpload')}</p>` : ''}
      ${state.uploads.slice(0, 24).map((u) => `
        <div class="cloth-item upload-item ${state.selectedUploads.includes(u.name) ? 'selected' : ''}" data-upload="${u.name}">
          <img src="${u.url}" alt="${u.name}" loading="lazy" />
          <span>${u.name.replace(/^upload_[^_]+_/, '').replace(/^src_[^_]+_/, '').slice(0, 20)}</span>
          <button class="btn ghost small del-upload" data-del-upload="${u.name}" title="${t('upl.del')}">x</button>
        </div>
      `).join('')}
    </div>
  `
}

function jobCards(jobs) {
  if (!jobs.length) return `<p class="hint">${t('queue.empty')}</p>`
  return jobs.map((j) => `
    <article class="job" data-id="${j.id}">
      <img class="${j.status === 'done' ? 'zoomable' : ''}" src="${j.preview_url || j.source_url}" alt="" data-preview="${j.status === 'done' ? j.id : ''}" />
      <div class="meta">
        <b>${escapeHtml(j.title || j.source_name)}</b>
        <small>${fmtAudience(j.audience) || j.audience} · ${escapeHtml(j.message)}</small>
        <div class="bar"><i style="width:${j.progress || 0}%"></i></div>
      </div>
      <div>
        <div class="status ${j.status}">${j.status}</div>
        <div class="actions">
          ${j.status === 'done' ? `<button class="btn ghost small" data-preview="${j.id}">${t('job.preview')}</button>
          <a class="btn ghost small" href="/api/download/${j.id}" data-dl="${j.id}">${t('job.download')}</a>
          <button class="btn ghost small" data-push="${j.id}">${j.pushed_stores?.length ? t('job.pushedCount', { n: j.pushed_stores.length }) : t('job.push')}</button>` : ''}
        </div>
      </div>
    </article>
  `).join('')
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
}

function bindLogin() {
  const langBtn = document.querySelector('#login-lang')
  if (langBtn) langBtn.onclick = () => { setLang(getLang() === 'zh' ? 'en' : 'zh'); render() }
  document.querySelector('#login').onclick = async () => {
    const err = document.querySelector('#login-err')
    err.textContent = ''
    try {
      const data = await api('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: document.querySelector('#u').value,
          password: document.querySelector('#p').value
        })
      })
      state.token = data.token
      state.user = data
      localStorage.setItem(TOKEN_KEY, data.token)
      await boot()
    } catch (e) {
      err.textContent = e.message
    }
  }
}

function bindDashboard() {
  const langSwitch = document.querySelector('#lang-switch')
  if (langSwitch) langSwitch.onclick = () => { setLang(getLang() === 'zh' ? 'en' : 'zh'); render() }
  document.querySelectorAll('.nav button[data-view]').forEach((b) => {
    b.onclick = () => { state.view = b.dataset.view; render() }
  })
  const logoutBtn = document.querySelector('#logout')
  if (logoutBtn) logoutBtn.onclick = logout
  const ethFilter = document.querySelector('#eth-filter')
  if (ethFilter) {
    ethFilter.onchange = (e) => { state.ethnicity = e.target.value; render() }
  }
  const ethManage = document.querySelector('#eth-manage')
  const ethBox = document.querySelector('#eth-manage-box')
  if (ethManage && ethBox) {
    ethManage.onclick = () => { ethBox.style.display = ethBox.style.display === 'none' ? 'block' : 'none' }
  }
  const addEth = document.querySelector('#add-eth')
  if (addEth) {
    addEth.onclick = async () => {
      const value = document.querySelector('#new-eth-value').value.trim()
      const label = document.querySelector('#new-eth-label').value.trim()
      if (!value || !label) { alert(t('eth.needBoth')); return }
      try {
        await api('/api/ethnicities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value, label })
        })
        document.querySelector('#new-eth-value').value = ''
        document.querySelector('#new-eth-label').value = ''
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  }
  document.querySelectorAll('.eth-rename').forEach((btn) => {
    btn.onclick = async () => {
      const value = btn.dataset.value
      const label = prompt(t('eth.renamePh'), btn.dataset.label)
      if (!label || label === btn.dataset.label) return
      try {
        await api(`/api/ethnicities/${encodeURIComponent(value)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label })
        })
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  document.querySelectorAll('.eth-del').forEach((btn) => {
    btn.onclick = async () => {
      const value = btn.dataset.value
      if (!confirm(t('eth.delConfirm', { v: btn.dataset.value }))) return
      try {
        await api(`/api/ethnicities/${encodeURIComponent(value)}`, { method: 'DELETE' })
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  document.querySelectorAll('.base').forEach((el) => {
    el.onclick = (e) => {
      if (e.target.dataset.del) return
      if (e.target.dataset.bsel) return
      const base = state.bases.find((b) => b.id === el.dataset.id)
      if (base && base.audience) state.audience = base.audience
      state.selectedBase = el.dataset.id
      render()
    }
  })
  document.querySelectorAll('[data-bsel]').forEach((cb) => {
    cb.onclick = (e) => e.stopPropagation()
    cb.onchange = () => {
      const id = cb.dataset.bsel
      if (cb.checked) {
        if (!state.selectedBases.includes(id)) state.selectedBases.push(id)
      } else {
        state.selectedBases = state.selectedBases.filter((x) => x !== id)
      }
      render()
    }
  })
  const baseBatchSel = document.querySelector('#base-batch-sel')
  const baseBatchBox = document.querySelector('#base-batch-box')
  if (baseBatchSel && baseBatchBox) {
    const countEl = document.querySelector('#base-batch-count')
    if (countEl) countEl.textContent = state.selectedBases.length ? t('base.selCount', { n: state.selectedBases.length }) : ''
    baseBatchSel.onclick = () => {
      state.baseBatchOpen = !state.baseBatchOpen
      render()
    }
    const baseBatchApply = document.querySelector('#base-batch-apply')
    baseBatchApply.onclick = async () => {
      if (!state.selectedBases.length) { alert(t('base.needSel')); return }
      const patch = {}
      const baud = document.querySelector('#bbaud').value
      const bage = document.querySelector('#bbage').value
      const bbeth = document.querySelector('#bbeth').value
      if (baud) patch.audience = baud
      if (bage) patch.age_band = bage
      if (bbeth) patch.ethnicity = bbeth
      if (!Object.keys(patch).length) { alert(t('base.needCat')); return }
      try {
        const res = await api('/api/bases/batch/patch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ids: state.selectedBases, patch })
        })
        if (!res.ok) throw new Error(res.detail || t('base.batchFail'))
        state.selectedBases = []
        document.querySelector('#bbaud').value = ''
        document.querySelector('#bbage').value = ''
        document.querySelector('#bbeth').value = ''
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  }
  document.querySelectorAll('[data-del]').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation()
      if (!confirm(t('base.retireConfirm'))) return
      try {
        await api(`/api/bases/${btn.dataset.del}`, { method: 'DELETE' })
        if (state.selectedBase === btn.dataset.del) state.selectedBase = ''
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  document.querySelectorAll('[data-preview]').forEach((el) => {
    if (!el.dataset.preview) return
    el.onclick = () => { state.previewJob = el.dataset.preview; render() }
  })
  const closePv = document.querySelector('#close-preview')
  const closePv2 = document.querySelector('#close-preview-2')
  if (closePv) closePv.onclick = () => { state.previewJob = ''; render() }
  if (closePv2) closePv2.onclick = () => { state.previewJob = ''; render() }
  const modal = document.querySelector('#preview-modal')
  if (modal) modal.onclick = (e) => { if (e.target === modal) { state.previewJob = ''; render() } }
  const drop = document.querySelector('#drop')
  const file = document.querySelector('#file')
  if (drop && file) {
    drop.onclick = () => file.click()
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('on') }
    drop.ondragleave = () => drop.classList.remove('on')
    drop.ondrop = (e) => {
      e.preventDefault()
      drop.classList.remove('on')
      addFiles(e.dataTransfer.files)
    }
    file.onchange = () => addFiles(file.files)
    const audience = document.querySelector('#audience')
    if (audience) audience.onchange = (e) => { state.audience = e.target.value }
    const gtype = document.querySelector('#gtype')
    if (gtype) gtype.onchange = (e) => { state.garmentType = e.target.value }
    const fmt = document.querySelector('#fmt')
    if (fmt) fmt.onchange = (e) => { state.outputFormat = e.target.value }
    const bg = document.querySelector('#bg')
    if (bg) bg.onchange = (e) => { state.background = e.target.value }
    const run = document.querySelector('#run')
    if (run) run.onclick = submitBatch
    const samples = document.querySelector('#samples')
    if (samples) samples.onclick = loadSamples
    const clear = document.querySelector('#clear')
    if (clear) clear.onclick = () => { state.files = []; state.selectedUploads = []; renderFileThumbs(); render() }
    const addSelected = document.querySelector('#add-selected')
    if (addSelected) {
      addSelected.onclick = async () => {
        for (const name of state.selectedUploads) {
          try {
            const upload = state.uploads.find(u => u.name === name)
            const url = upload ? upload.url : `/files/uploads/${name}`
            const res = await fetch(url)
            const blob = await res.blob()
            state.files.push(new File([blob], name, { type: blob.type || 'image/jpeg' }))
          } catch (e) {
            console.error(t('job.loadedErr'), name, e)
          }
        }
        state.selectedUploads = []
        renderFileThumbs()
        const run = document.querySelector('#run')
        if (run) run.textContent = t('btn.generate', { n: state.files.length })
        render()
      }
    }
    renderFileThumbs()
  }
  const addBase = document.querySelector('#add-base')
  if (addBase) {
    addBase.onclick = saveBase
  }
  document.querySelectorAll('[data-cloth]').forEach((item) => {
    item.onclick = async () => {
      const name = item.dataset.cloth
      try {
        const res = await fetch(`/examples/cloth/${name}`)
        const blob = await res.blob()
        state.files.push(new File([blob], name, { type: blob.type || 'image/jpeg' }))
        renderFileThumbs()
        const run = document.querySelector('#run')
        if (run) run.textContent = t('btn.generate', { n: state.files.length })
      } catch (e) {
        console.error(t('job.loadSampleErr'), e)
      }
    }
  })
  document.querySelectorAll('[data-upload]').forEach((item) => {
    item.onclick = async (e) => {
      if (e.target.dataset.delUpload) return
      const name = item.dataset.upload
      if (state.selectedUploads.includes(name)) {
        state.selectedUploads = state.selectedUploads.filter(n => n !== name)
      } else {
        state.selectedUploads.push(name)
      }
      render()
    }
  })
  const catFilter = document.querySelector('#cat-filter')
  if (catFilter) {
    catFilter.onchange = () => {
      const cat = catFilter.value
      if (!cat) {
        refreshUploads()
        render()
      } else {
        api(`/api/uploads?category=${encodeURIComponent(cat)}`).then(data => {
          state.uploads = data
          render()
        })
      }
    }
  }
  const catManage = document.querySelector('#cat-manage')
  const catBox = document.querySelector('#cat-manage-box')
  if (catManage && catBox) {
    catManage.onclick = () => { catBox.style.display = catBox.style.display === 'none' ? 'block' : 'none' }
  }
  const addCat = document.querySelector('#add-cat')
  if (addCat) {
    addCat.onclick = async () => {
      const name = document.querySelector('#new-cat-name').value.trim()
      if (!name) { alert(t('upl.nameRequired')); return }
      try {
        await api('/api/uploads/categories', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name })
        })
        document.querySelector('#new-cat-name').value = ''
        await refreshUploads()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  }
  document.querySelectorAll('.cat-rename').forEach((btn) => {
    btn.onclick = async () => {
      const cat = btn.dataset.cat
      const name = prompt(t('upl.newName'), cat)
      if (!name || name === cat) return
      try {
        await api(`/api/uploads/categories/${encodeURIComponent(cat)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name })
        })
        await refreshUploads()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  document.querySelectorAll('.cat-del').forEach((btn) => {
    btn.onclick = async () => {
      const cat = btn.dataset.cat
      if (!confirm(t('upl.delCatConfirm', { c: cat }))) return
      try {
        await api(`/api/uploads/categories/${encodeURIComponent(cat)}`, { method: 'DELETE' })
        await refreshUploads()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  const catMove = document.querySelector('#cat-move')
  const catMoveSelect = document.querySelector('#cat-move-select')
  const catName = document.querySelector('#cat-name')
  if (catMove) {
    catMove.onclick = () => {
      if (state.selectedUploads.length === 0) {
        alert(t('upl.moveHint'))
        return
      }
      const show = catMoveSelect.style.display !== 'inline-block'
      catMoveSelect.style.display = show ? 'inline-block' : 'none'
      catName.style.display = 'none'
      catMoveSelect.value = ''
    }
    catMoveSelect.onchange = () => {
      const val = catMoveSelect.value
      if (!val) return
      if (val === '__new__') {
        catName.style.display = 'inline-block'
        catName.focus()
        return
      }
      applyCategory(val)
    }
    const applyCategory = async (category) => {
      if (!category || state.selectedUploads.length === 0) return
      try {
        for (const name of state.selectedUploads) {
          await api(`/api/uploads/${name}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category })
          })
        }
        state.selectedUploads = []
        catName.value = ''
        catName.style.display = 'none'
        catMoveSelect.style.display = 'none'
        catMoveSelect.value = ''
        await refreshUploads()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
    catName.onkeydown = async (e) => {
      if (e.key === 'Enter' && catName.value && state.selectedUploads.length) {
        const category = catName.value.trim()
        try {
          const existing = await api('/api/uploads/categories')
          if (!existing.includes(category)) {
            await api('/api/uploads/categories', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: category })
            })
          }
          await applyCategory(category)
        } catch (err) {
          alert(err.message)
        }
      }
      if (e.key === 'Escape') {
        catName.value = ''
        catName.style.display = 'none'
        render()
      }
    }
  }
  const uploadAdd = document.querySelector('#upload-lib-add')
  const uploadFile = document.querySelector('#upload-lib-file')
  if (uploadAdd && uploadFile) {
    uploadAdd.onclick = () => uploadFile.click()
    uploadFile.onchange = async () => {
      for (const f of uploadFile.files) {
        try {
          const form = new FormData()
          form.append('file', f, f.name)
          const res = await fetch('/api/uploads', {
            method: 'POST',
            headers: { Authorization: `Bearer ${state.token}` },
            body: form
          })
          if (!res.ok) throw new Error(t('upl.uploadFail'))
        } catch (e) {
          alert(t('upl.uploadFileFail', { n: f.name, msg: e.message }))
        }
      }
      uploadFile.value = ''
      await refreshUploads()
      render()
    }
  }
  document.querySelectorAll('[data-del-upload]').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation()
      if (!confirm(t('upl.delConfirm'))) return
      try {
        await api(`/api/uploads/${btn.dataset.delUpload}`, { method: 'DELETE' })
        await refreshUploads()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  document.querySelectorAll('[data-push]').forEach((btn) => {
    btn.onclick = () => { state.pushJob = btn.dataset.push; render() }
  })
  const closePU = document.querySelector('#close-push')
  const closePU2 = document.querySelector('#close-push-2')
  if (closePU) closePU.onclick = () => { state.pushJob = ''; render() }
  if (closePU2) closePU2.onclick = () => { state.pushJob = ''; render() }
  const pushModal = document.querySelector('#push-modal')
  if (pushModal) pushModal.onclick = (e) => { if (e.target === pushModal) { state.pushJob = ''; render() } }
  const confirmPush = document.querySelector('#confirm-push')
  if (confirmPush) {
    confirmPush.onclick = async () => {
      const ids = Array.from(document.querySelectorAll('#push-modal [data-store]:checked')).map((el) => el.dataset.store)
      const productIdEl = document.querySelector('#push-product-id')
      const productId = productIdEl ? productIdEl.value.trim() : ''
      try {
        await api(`/api/jobs/${state.pushJob}/push`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sku: '', product_url: '', product_id: productId, store_ids: ids })
        })
        const job = state.jobs.find((j) => j.id === state.pushJob)
        if (job) { job.message = t('push.updated'); job.pushed_stores = ids }
        state.pushJob = ''
        await refreshJobs()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  }
  document.querySelectorAll('[data-dl]').forEach((a) => {
    a.onclick = async (e) => {
      e.preventDefault()
      const res = await fetch(a.getAttribute('href'), { headers: { Authorization: `Bearer ${state.token}` } })
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = a.dataset.dl
      link.click()
    }
  })
  const dlAll = document.querySelector('#dl-all')
  if (dlAll) dlAll.onclick = downloadAll
  document.querySelectorAll('[data-del-store]').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation()
      if (!confirm(t('embed.delConfirm'))) return
      try {
        await api(`/api/stores/${btn.dataset.delStore}`, { method: 'DELETE' })
        await refreshStores()
        render()
      } catch (err) {
        alert(err.message)
      }
    }
  })
  const addStore = document.querySelector('#add-store')
  if (addStore) {
    addStore.onclick = async () => {
      const err = document.querySelector('#store-err')
      err.textContent = ''
      const name = document.querySelector('#sname').value.trim()
      if (!name) { err.textContent = t('embed.nameRequired'); return }
      try {
        await api('/api/stores', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            platform: document.querySelector('#splatform').value,
            shop_url: document.querySelector('#sshop').value.trim(),
            webhook_url: '',
            api_key: document.querySelector('#sapi').value.trim(),
            product_id: document.querySelector('#sprod').value.trim()
          })
        })
        document.querySelector('#sname').value = ''
        document.querySelector('#sapi').value = ''
        document.querySelector('#sprod').value = ''
        await refreshStores()
        render()
      } catch (e) {
        err.textContent = e.message
      }
    }
  }
}

function addFiles(list) {
  state.files = [...state.files, ...Array.from(list)]
  renderFileThumbs()
}

function renderFileThumbs() {
  const box = document.querySelector('#file-thumbs')
  const run = document.querySelector('#run')
  if (!box) return
  box.innerHTML = state.files.slice(0, 24).map((f) => `<img src="${URL.createObjectURL(f)}" alt="${f.name}">`).join('')
  if (run) run.textContent = t('btn.generate', { n: state.files.length })
}

async function loadSamples() {
  const files = []
  for (const name of CLOTH_NAMES) {
    const res = await fetch(`/examples/cloth/${name}`)
    const blob = await res.blob()
    files.push(new File([blob], name, { type: blob.type || 'image/jpeg' }))
  }
  state.files = files
  renderFileThumbs()
}

async function submitBatch() {
  const err = document.querySelector('#batch-err')
  err.textContent = ''
  if (!state.selectedBase) {
    err.textContent = t('err.selectedBase')
    return
  }
  if (!state.files.length) {
    err.textContent = t('err.dropFile')
    return
  }
  const form = new FormData()
  form.append('base_id', state.selectedBase)
  form.append('audience', state.audience)
  form.append('garment_type', state.garmentType)
  form.append('output_format', state.outputFormat)
  form.append('background', state.background)
  state.files.forEach((f) => form.append('files', f, f.name))
  try {
    await api('/api/jobs/batch', { method: 'POST', body: form })
    state.files = []
    state.view = 'queue'
    await refreshJobs()
    render()
  } catch (e) {
    err.textContent = e.message
  }
}

async function saveBase() {
  const err = document.querySelector('#base-err')
  err.textContent = ''
  const file = document.querySelector('#bfile').files[0]
  if (!file) {
    err.textContent = t('err.baseFile')
    return
  }
  const form = new FormData()
  form.append('name', document.querySelector('#bname').value || file.name)
  form.append('audience', document.querySelector('#baud').value)
  form.append('age_band', document.querySelector('#bage').value)
  form.append('ethnicity', document.querySelector('#beth').value)
  form.append('notes', document.querySelector('#bnotes').value)
  form.append('file', file)
  try {
    const created = await api('/api/bases', { method: 'POST', body: form })
    state.selectedBase = created.id
    await refreshJobs()
    render()
  } catch (e) {
    err.textContent = e.message
  }
}

async function downloadAll() {
  for (const job of state.jobs.filter((j) => j.status === 'done')) {
    const res = await fetch(`/api/download/${job.id}`, { headers: { Authorization: `Bearer ${state.token}` } })
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${job.title || job.id}.jpg`
    a.click()
  }
}

async function refreshJobs() {
  const [bases, jobs, stats, settings, me, ethnicities] = await Promise.all([
    api('/api/bases'),
    api('/api/jobs'),
    api('/api/stats'),
    api('/api/settings'),
    api('/api/me'),
    api('/api/ethnicities').catch(() => [])
  ])
  state.bases = bases
  state.jobs = jobs
  state.stats = stats
  state.settings = settings
  state.user = me
  state.ethnicities = ethnicities
  if (!state.selectedBase && bases[0]) state.selectedBase = bases[0].id
}

async function refreshUploads() {
  try {
    const [uploads, categories] = await Promise.all([
      api('/api/uploads'),
      api('/api/uploads/categories')
    ])
    state.uploads = uploads
    state.uploadCategories = categories
  } catch {}
}

async function refreshStores() {
  try {
    state.stores = await api('/api/stores')
  } catch {}
}

async function boot() {
  await refreshJobs()
  await refreshUploads()
  await refreshStores()
  try {
    state.license = await api('/api/license')
  } catch {}
  const params = new URLSearchParams(location.search)
  state.storeCode = params.get('store') || ''
  render()
  setInterval(async () => {
    if (!state.token) return
    try {
      await refreshJobs()
      await refreshUploads()
      if (state.view === 'queue') render()
    } catch {}
  }, 5000)
}

if (state.token) boot()
else render()
