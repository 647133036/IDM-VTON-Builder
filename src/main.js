import './style.css'

const TOKEN_KEY = 'atelier_token'
const AUDIENCE_LABEL = {
  women: '女装',
  men: '男装',
  girls: '女童',
  boys: '男童',
  baby: '婴幼儿',
  auto: '自动识别'
}

const state = {
  token: localStorage.getItem(TOKEN_KEY) || '',
  user: null,
  view: 'batch',
  bases: [],
  jobs: [],
  stats: {},
  settings: {},
  selectedBase: '',
  files: [],
  audience: 'auto',
  garmentType: 'auto',
  outputFormat: 'jpg',
  background: 'white',
  error: '',
  previewJob: '',
  license: null
}

async function api(path, options = {}) {
  const headers = options.headers || {}
  if (state.token) headers.Authorization = `Bearer ${state.token}`
  const res = await fetch(path, { ...options, headers })
  if (res.status === 401) {
    logout()
    throw new Error('需要登录')
  }
  if (!res.ok) {
    let detail = '请求失败'
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
        <h1>Atelier</h1>
        <p>店铺后台 · 固定底图换衣工作台。账号 admin / admin123，或 staff / staff123。</p>
        <label>账号<input id="u" value="admin" /></label>
        <label>密码<input id="p" type="password" value="admin123" /></label>
        <button class="btn gold" id="login">进入后台</button>
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
          <h1>Atelier Studio</h1>
          <span>${state.user?.name || 'Store Admin'} · 固定底图 + 换衣</span>
        </div>
        <nav class="nav">
          <button data-view="batch" class="${state.view === 'batch' ? 'on' : ''}">批量换衣</button>
          <button data-view="bases" class="${state.view === 'bases' ? 'on' : ''}">模特底座</button>
          <button data-view="queue" class="${state.view === 'queue' ? 'on' : ''}">任务队列</button>
          <button data-view="embed" class="${state.view === 'embed' ? 'on' : ''}">店铺嵌入</button>
          <button id="logout" class="${''}">退出</button>
        </nav>
      </header>
      ${licenseBanner()}
      <section class="kpis">
        ${kpi('队列', state.stats.queued)}
        ${kpi('生成中', state.stats.running)}
        ${kpi('已完成', state.stats.done)}
        ${kpi('失败', state.stats.failed)}
        ${kpi('底座', state.stats.bases)}
      </section>
      ${state.view === 'batch' ? batchView() : ''}
      ${state.view === 'bases' ? basesView() : ''}
      ${state.view === 'queue' ? queueView() : ''}
      ${state.view === 'embed' ? embedView() : ''}
      ${state.previewJob ? previewModal() : ''}
    </div>
  `
}

function licenseBanner() {
  if (!state.license) return ''
  const l = state.license
  if (l.license_mode === 'commercial') {
    return `<div class="banner warn"><strong>商业模式</strong>：IDM-VTON 已禁用（CC BY-NC-SA 禁止商用），当前使用 ${l.current_engine}。请使用商用模型或 API。</div>`
  }
  if (l.license_mode === 'research') {
    if (l.idm_vton_available) {
      return `<div class="banner info"><strong>研究模式</strong>：IDM-VTON 可用，仅限测试/学习，不得用于生产。详见 <a href="/docs/LICENSE_COMPLIANCE.md" target="_blank">合规说明</a>。</div>`
    }
    return `<div class="banner info"><strong>研究模式</strong>：IDM-VTON 不可用（缺少 GPU 或权重），当前使用 ${l.current_engine}。</div>`
  }
  return `<div class="banner warn"><strong>未设置 LICENSE_MODE</strong>：建议设置 research（测试）或 commercial（生产）。详见 <a href="/docs/LICENSE_COMPLIANCE.md" target="_blank">合规说明</a>。</div>`
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
            <h3>成品大图预览</h3>
            <small>${escapeHtml(job.title || job.source_name)} · ${AUDIENCE_LABEL[job.audience] || job.audience} · ${escapeHtml(job.garment_type)}</small>
          </div>
          <button class="btn ghost small" id="close-preview">关闭</button>
        </div>
        <div class="modal-body">
          <div class="preview-col">
            <span class="preview-label">原服装图</span>
            <img class="preview-img" src="${job.source_url}" alt="原服装" />
          </div>
          <div class="preview-col main">
            <span class="preview-label">成品 · ${base ? escapeHtml(base.name) : '底座'}</span>
            <img class="preview-img big" src="${job.result_url}" alt="成品" />
          </div>
        </div>
        <div class="modal-foot">
          <a class="btn gold" href="/api/download/${job.id}" download>下载 ${job.result_url.split('.').pop().toUpperCase()}</a>
          <button class="btn ghost" id="close-preview-2">关闭</button>
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
        <h3>1. 选择固定底图</h3>
        <div class="bases" id="base-list">${baseCards()}</div>
        <h3 style="margin-top:16px">2. 品类与输出</h3>
        <div class="row">
          <label>受众
            <select id="audience">
              ${Object.entries(AUDIENCE_LABEL).map(([k, v]) => `<option value="${k}" ${state.audience === k ? 'selected' : ''}>${v}</option>`).join('')}
            </select>
          </label>
          <label>服装
            <select id="gtype">
              <option value="auto">自动</option>
              <option value="upper">上衣</option>
              <option value="dress">裙装</option>
              <option value="lower">下装</option>
              <option value="onesie">连体</option>
            </select>
          </label>
        </div>
        <div class="row">
          <label>格式
            <select id="fmt">
              <option value="jpg">JPEG 白底</option>
              <option value="png">PNG</option>
            </select>
          </label>
          <label>背景
            <select id="bg">
              <option value="white">白底</option>
              <option value="transparent">透明</option>
            </select>
          </label>
        </div>
        <p class="hint">每件只出一张正面成品，长边 2048px。女装 / 男装 / 女童 / 男童 / 婴幼儿底座可随时更换。</p>
      </aside>
      <section class="panel">
        <h3>3. 拖入平铺图或幽灵模特图</h3>
        <div class="drop" id="drop">把几十到上百张服装图拖到这里，或点击选择文件<br><small>支持 JPG / PNG / WEBP，可一次多选</small></div>
        <input id="file" type="file" accept="image/*" multiple hidden />
        <div class="thumbs" id="file-thumbs"></div>
        <div class="row" style="margin-top:12px">
          <button class="btn gold" id="run">一键生成 ${state.files.length} 件</button>
          <button class="btn ghost" id="samples">载入示例平铺图</button>
          <button class="btn ghost" id="clear">清空</button>
        </div>
        <p class="err" id="batch-err">${state.error}</p>
        <h3 style="margin-top:18px">示例服装库 · 点击加入</h3>
        <div class="cloth-lib" id="cloth-lib">${clothLibrary()}</div>
        <h3 style="margin-top:18px">最近任务</h3>
        <div class="queue">${jobCards(state.jobs.slice(0, 8))}</div>
      </section>
    </div>
  `
}

function basesView() {
  return `
    <div class="grid">
      <section class="panel">
        <h3>当前模特底座</h3>
        <p class="hint">底座是固定前视人台。换衣时只替换服装区域，脸、手、姿态和背景保持不变。可上传自家模特替换默认底座。任务选「自动识别」时，受众优先取所选手底的受众标签。</p>
        <div class="bases">${state.bases.map((b) => `
          <div class="base ${state.selectedBase === b.id ? 'sel' : ''}" data-id="${b.id}">
            <img src="${b.url}" alt="${escapeHtml(b.name)}" />
            <em>${escapeHtml(b.name)}</em>
            <button class="btn ghost small" data-del="${b.id}" title="下架此底座">下架</button>
          </div>`).join('')}</div>
      </section>
      <section class="panel">
        <h3>上传新底座</h3>
        <label>名称<input id="bname" placeholder="例如 女装·亚洲模特 B" /></label>
        <div class="row">
          <label>受众
            <select id="baud">
              <option value="women">女装</option>
              <option value="men">男装</option>
              <option value="girls">女童</option>
              <option value="boys">男童</option>
              <option value="baby">婴幼儿</option>
            </select>
          </label>
          <label>年龄带
            <select id="bage">
              <option value="adult">成人</option>
              <option value="kids">儿童</option>
              <option value="baby">婴幼儿</option>
            </select>
          </label>
        </div>
        <label>备注<input id="bnotes" placeholder="站姿、适合品类" /></label>
        <label>底图文件<input id="bfile" type="file" accept="image/*" /></label>
        <button class="btn gold" id="add-base">保存底座</button>
        <p class="err" id="base-err"></p>
      </section>
    </div>
  `
}

function queueView() {
  const done = state.jobs.filter((j) => j.status === 'done')
  return `
    <section class="panel">
      <h3>生成队列</h3>
      <div class="row" style="margin-bottom:12px">
        <button class="btn ghost small" id="dl-all">打包下载已完成 ${done.length} 张</button>
      </div>
      <div class="queue">${jobCards(state.jobs)}</div>
    </section>
  `
}

function embedView() {
  const snippet = `<iframe src="${location.origin}/embed.html" style="width:100%;min-height:720px;border:0;border-radius:16px"></iframe>`
  return `
    <div class="grid">
      <section class="panel">
        <h3>嵌进店铺后台</h3>
        <p class="hint">前端是独立 SPA，用 iframe 或把 /embed.html 挂到现有 admin 路由即可。后端 FastAPI 部署在你的云账号，数据与算力不出域。</p>
        <div class="code">${escapeHtml(snippet)}</div>
        <p class="hint">生产环境把 JWT 密钥、管理员密码和 GPU 推理服务配到环境变量。当前演示引擎是服务端合成流水线，接口与 IDM-VTON 任务队列对齐，可替换为 GPU worker。</p>
      </section>
      <section class="panel">
        <h3>API</h3>
        <div class="code">POST /api/auth/login
GET  /api/bases
POST /api/jobs/batch
GET  /api/jobs
POST /api/jobs/{id}/push
GET  /api/download/{id}</div>
      </section>
    </div>
  `
}

function baseCards() {
  return state.bases.map((b) => `
    <div class="base ${state.selectedBase === b.id ? 'sel' : ''}" data-id="${b.id}">
      <img src="${b.url}" alt="${escapeHtml(b.name)}" />
      <em>${AUDIENCE_LABEL[b.audience] || escapeHtml(b.name)}</em>
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

function jobCards(jobs) {
  if (!jobs.length) return '<p class="hint">暂无任务</p>'
  return jobs.map((j) => `
    <article class="job" data-id="${j.id}">
      <img class="${j.status === 'done' ? 'zoomable' : ''}" src="${j.preview_url || j.source_url}" alt="" data-preview="${j.status === 'done' ? j.id : ''}" />
      <div class="meta">
        <b>${escapeHtml(j.title || j.source_name)}</b>
        <small>${AUDIENCE_LABEL[j.audience] || j.audience} · ${escapeHtml(j.message)}</small>
        <div class="bar"><i style="width:${j.progress || 0}%"></i></div>
      </div>
      <div>
        <div class="status ${j.status}">${j.status}</div>
        <div class="actions">
          ${j.status === 'done' ? `<button class="btn ghost small" data-preview="${j.id}">大图预览</button>
          <a class="btn ghost small" href="/api/download/${j.id}" data-dl="${j.id}">下载</a>
          <button class="btn ghost small" data-push="${j.id}">推送到商品页</button>` : ''}
        </div>
      </div>
    </article>
  `).join('')
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
}

function bindLogin() {
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
  document.querySelectorAll('.nav button[data-view]').forEach((b) => {
    b.onclick = () => { state.view = b.dataset.view; render() }
  })
  document.querySelector('#logout').onclick = logout
  document.querySelectorAll('.base').forEach((el) => {
    el.onclick = (e) => {
      if (e.target.dataset.del) return
      const base = state.bases.find((b) => b.id === el.dataset.id)
      if (base && base.audience) state.audience = base.audience
      state.selectedBase = el.dataset.id
      render()
    }
  })
  document.querySelectorAll('[data-del]').forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation()
      if (!confirm('确认下架此底座？已生成的任务不受影响。')) return
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
    document.querySelector('#audience').onchange = (e) => { state.audience = e.target.value }
    document.querySelector('#gtype').onchange = (e) => { state.garmentType = e.target.value }
    document.querySelector('#fmt').onchange = (e) => { state.outputFormat = e.target.value }
    document.querySelector('#bg').onchange = (e) => { state.background = e.target.value }
    document.querySelector('#run').onclick = submitBatch
    document.querySelector('#samples').onclick = loadSamples
    document.querySelector('#clear').onclick = () => { state.files = []; renderFileThumbs() }
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
        if (run) run.textContent = `一键生成 ${state.files.length} 件`
      } catch (e) {
        console.error('加载示例服装失败', e)
      }
    }
  })
  document.querySelectorAll('[data-push]').forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/jobs/${btn.dataset.push}/push`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sku: '', product_url: '/products/staging' })
      })
      await refreshJobs()
      render()
    }
  })
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
  if (run) run.textContent = `一键生成 ${state.files.length} 件`
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
    err.textContent = '请先选择模特底座'
    return
  }
  if (!state.files.length) {
    err.textContent = '请拖入服装图'
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
    err.textContent = '请选择底图'
    return
  }
  const form = new FormData()
  form.append('name', document.querySelector('#bname').value || file.name)
  form.append('audience', document.querySelector('#baud').value)
  form.append('age_band', document.querySelector('#bage').value)
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
  const [bases, jobs, stats, settings, me] = await Promise.all([
    api('/api/bases'),
    api('/api/jobs'),
    api('/api/stats'),
    api('/api/settings'),
    api('/api/me')
  ])
  state.bases = bases
  state.jobs = jobs
  state.stats = stats
  state.settings = settings
  state.user = me
  if (!state.selectedBase && bases[0]) state.selectedBase = bases[0].id
}

async function boot() {
  await refreshJobs()
  try {
    state.license = await api('/api/license')
  } catch {}
  render()
  setInterval(async () => {
    if (!state.token) return
    try {
      await refreshJobs()
      if (state.view === 'queue') render()
    } catch {}
  }, 2500)
}

if (state.token) boot()
else render()
