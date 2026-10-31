# Atelier v0.1.4

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

## v0.1.4 更新日志

本轮版本解决三个出图致命缺陷：**换图案**（SDXL 不按原衣图案画）、**衣袖截断**（长袖只画到腰）、**胸口深 V / 肩膀重影**（脸区遮罩吞掉胸口、高频纹理在肩臂交界叠成重影）。同时引入商业级换衣流程的动态遮罩与无缝贴回。

### 解决了什么问题

| 缺陷 | 根因 | 修复 |
|------|------|------|
| **换图案** | Canny 0.35 太低，衣服图案边缘没参与生成；IP-Adapter 0.8 对复杂花纹不够强 | Canny 按品类自动调权（实心面料 0.55 / 下装 0.45），IP-Adapter 提到 0.9，reapply texture_strength 提到 0.70 |
| **衣袖截断** | `fit_garment_to_body` 把衣高 clamp 到躯干框 1.6 倍，canvas 只有躯干高，袖子被裁 | 去掉 max_scale 高度 clamp，canvas 高度取 `max(th, nh+pad)` 容纳长袖，`shirt_mask` 用 fitted 实际尺寸 |
| **胸口深 V** | `clothing_mask` 的 `face_margin=0.65` 四方向等扩，脸框向下吞掉脖子和胸口 | `face_margin` 降到 0.35，新增 `down_margin_frac=0.08` 只向下扩到下巴线，脖子和胸口留给换衣区 |
| **肩膀重影** | `reapply_garment_detail` 把高频纹理均匀贴到整个 mask 区，肩臂交界对齐偏差叠加成重影 | 加入 distanceTransform 边缘衰减：mask 中心区全强度贴纹理，边缘 25% 区渐弱到 0 |

### 优化了什么

- **ControlNet Canny 按品类自动调权**：实心面料（upper/dress/onesie）0.55 锁图案结构；下装 0.45 防鬼影。`VTON_CANNY` 环境变量仍可显式覆盖。
- **IP-Adapter 默认 0.9**：从 0.8 提升，blend_tryon 身份锁已保护人脸，可以加大服装特征注入权重。`VTON_IPA` 仍可覆盖。
- **reapply_garment_detail 边缘衰减**：用 `cv2.distanceTransform` 计算每个像素到 mask 边缘的距离，在 0-25% 距离区间线性衰减 texture_strength，肩臂交界不再叠高频重影。
- **reapply texture_strength 0.70**：从 0.55 提升，高频纹理回贴更狠，补 1024 分辨率丢失的细节。
- **face_margin 非对称扩展**：`_detect_face_mask` 新增 `down_margin_frac` 参数，上/左/右扩 0.35 保护额头耳侧，向下只扩 0.08 盖到下巴线。

### 增加了什么

- **`detect_garment_extent`（服装属性检测）**：从平铺衣料图自动判断袖长和衣长。按内容 alpha/背景差分取 bbox，纵向三段（肩/胸/下摆）宽度比 + 长宽比判定 `sleeve: long|short|sleeveless` 和 `hem: waist|long`。纯 CPU，无需模型推理。
- **动态 `clothing_mask`（商业级遮罩）**：按检测结果动态调整换衣区范围：
  - 长袖 → mask 延伸到 88% 身高（含整条手臂，新衣服的袖子能画上去）
  - 短袖 → 62%（只含大臂）
  - 无袖 → 55%（不含手臂）
  - 连衣裙 → 按衣长决定（过腰 96% / 正常 82%）
- **`blend_tryon` 升级为 SeamlessClone**：身份区（mask=0，脸/手/脚）按连通分量逐块用 `cv2.seamlessClone(NORMAL_CLONE)` 无缝贴回原图；质心落在孔洞外时自动取组件内中位点；衣服区用增强 alpha 混合（×3 权重）锁死生成结果；seamlessClone 失败时安全回退羽化混合。

### 实现了什么功能

商业级换衣流程三步落地：

```
1. 原图剥离    detect_garment_extent → 分析衣料袖长/衣长
                clothing_mask        → 动态遮罩:长袖含手臂,无袖不含
                
2. AI 重绘     SDXL + ControlNet(Canny 0.55 锁图案) + IP-Adapter(0.9 注入服装特征)
                reapply_garment_detail → 高频纹理边缘衰减贴回(中心强/边缘弱)
                
3. 无缝贴回    blend_tryon → seamlessClone 逐块贴回脸/手/脚(无硬边)
```

## v0.1.3 更新日志（历史）

v0.1.3 引入「AI 出骨架光影，代码保商品细节」混合生成策略：`align_garment_to_pose` 几何对齐 + `reapply_garment_detail` 纹理颜色回贴 + `blend_tryon` 身份锁。全部纯 CPU 后处理，无需重训即可生效。同时修复 LoRA 权重加载权威性（PEFT 优先）、训练几何（time_ids / letterbox / VAE fp32 / min-SNR）、推理默认值（VTON_AR / VTON_LORA）。

## 功能

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 批量拖入几十到上百张服装图，只出正面，长边 2048px
- 白底 JPEG 或透明 PNG
- 已上传图片形成可复用图片库，支持服饰分类管理与批量调整分类
- 底座分类（受众 / 年龄带 / 人种）可管理、可批量调整
- 人种标签可自定义（默认 10 个地理区域），换衣引擎按底座人种修正体型（肩宽、躯干比例）
- 多店铺管理：每店铺独立 embed 嵌入链接 + CSV/SKU 批量推送
- 真实推送：Shopify Admin API 上传商品图，WordPress REST 媒体 + WooCommerce 商品关联
- 成品大图预览弹窗，左右对照原图与成品
- 界面中英文一键切换，语言偏好本地持久化
- 三层引擎体系：IDM-VTON（需授权）/ SDXL（商用合规）/ 合成引擎（无 GPU 兜底）
- **v0.1.4 新增**：动态袖长检测 + 动态遮罩 + SeamlessClone 无缝贴回 + Canny/IP-Adapter/texture 权重提升

## 安全

本项目经过两轮安全审查。当前状态如下：

### 已修复

- **文件服务鉴权**：`/files/{path}` 无需登录，但仅暴露 `uploads` / `results` / `bases` 三个白名单目录，仅允许图片后缀，内部文件返回 404；响应带 `X-Content-Type-Options: nosniff`
- **上传路径校验**：文件名和分类名通过正则白名单校验，拒绝路径穿越
- **图片内容验证**：PIL 解码校验，仅接受 PNG / JPEG / WEBP，最大 20MB
- **JWT 密钥**：无硬编码默认值；未设置时随机生成并持久化（chmod 600），多 worker 共享
- **账号密码**：无公开默认密码；未设置时随机生成并输出到 stdout
- **角色权限**：admin 可执行所有操作，staff 仅可创建任务和查看
- **凭据脱敏**：`/api/stores` 响应不返回明文 api_key
- **SSRF 防护**：`/api/push` 强制 https + `.myshopify.com`，拒绝内网地址，`allow_redirects=False`
- **XSS 防护**：前端 `escapeHtml` 转义；服务端枚举白名单校验
- **GPU 并发**：扩散推理路径全局 `threading.Lock` 串行化
- **SDXL 管线缓存**：构建一次后常驻
- **批量事务性**：`/api/jobs/batch` 先全量校验再统一写盘

### 已知限制

- DNS 重绑定（TOCTOU）未完全消除
- 无 GPU 环境下合成引擎为启发式近似，效果有限
- 去水印为浅色角部 best-effort

## 图片规格

**上传**：PNG / JPEG / WEBP，最大 20MB，任意尺寸

**生成处理（v0.1.4 商业级流程）**：
1. `detect_garment_extent` 分析衣料袖长/衣长
2. `crop_garment_to_content` 裁剪主体
3. `align_garment_to_pose` 几何对齐送 IP-Adapter
4. ControlNet（OpenPose 0.8 + Canny 0.55 锁图案 + Depth 0.95）+ IP-Adapter 0.9
5. SDXL 生成姿态/光影/人体
6. `reapply_garment_detail` 边缘衰减贴回纹理 + Reinhard 颜色迁移（texture_strength 0.70）
7. `blend_tryon` 动态 mask + SeamlessClone 无缝贴回脸/手/脚
8. `force_white_background` 纯白底

**输出**：长边 2048px，宽高比跟随底座图

## 技术栈

**前端**：原生 JavaScript（ES 模块）+ Vite 5 + 自研 i18n（zh/en）

**后端**：Python 3 + FastAPI + Uvicorn + PyJWT + PIL / NumPy / OpenCV + JSON 文件持久化

**AI 引擎**：
- IDM-VTON（CC BY-NC-SA 4.0，禁止商用）
- SDXL + ControlNet + IP-Adapter（商用合规）
- 合成引擎（无 GPU 兜底）

**训练**：SDXL LoRA（双格式保存 PEFT + diffusers）+ ControlNet + 一键脚本 `MODEL=lora bash run_train.sh`

## 快速启动（无 GPU）

```bash
# 后端
pip3 install --break-system-packages -r backend/requirements.txt
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 前端
npm install
npm run dev
```

首次启动时后端为 `admin` 和 `staff` 生成随机密码并输出到日志。生产部署：

```bash
export STORE_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export STORE_ADMIN_PASSWORD="your-secure-password"
export STORE_STAFF_PASSWORD="your-secure-password"
```

## GPU 部署（商用模式）

```bash
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
git submodule update --init --recursive --depth 1
bash scripts/download_sdxl_weights.sh
cd backend
pip install -r requirements.txt
pip install "torch>=2.1.0" "diffusers>=0.27.0" "transformers>=4.40.0" "accelerate>=0.30.0" controlnet-aux rembg
export LICENSE_MODE=commercial
export STORE_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export STORE_ADMIN_PASSWORD="your-secure-password"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/license
```

## SDXL 训练

```bash
cd training
MODEL=lora bash run_train.sh         # LoRA（约 11 分钟 / 800 步）
MODEL=controlnet bash run_train.sh   # ControlNet
MODEL=lora_family bash run_train.sh  # 服装 + 人像双 LoRA
```

训练基线（v0.1.3+）：
- `bfloat16`（Ampere+ GPU），CPU 回退 `float32`
- 双格式保存：PEFT `adapter_model.safetensors` + diffusers `pytorch_lora_weights.safetensors`
- `time_ids` `[res,res,0,0,res,res]`、`letterbox_rgb` 保宽高比、VAE fp32、min-SNR gamma=5.0

实测版本：PyTorch 2.13.0 + cu130 / diffusers 0.40.0（A10 23GB 验证）

## 多店铺与推送

### CSV/SKU 批量推送（v0.1.4）

- 任务队列页「导出 CSV+图」按钮：POST `/api/jobs/export-csv` 下载 ZIP（CSV 清单 + 成品图）
- 店铺卡片「批量推送」按钮：上传 CSV/ZIP，POST `/api/stores/{id}/batch-push` 批量推送
- 替换图优先于原成品，SKU 回退链：CSV 行 sku > 店铺 default_sku > 空
- 店铺表单新增 `default_sku` 字段
- URL 参数 `?store=xxx` 自动绑定店铺并切换到换衣视图

### 单张推送

完成的任务勾选店铺推送（admin 权限）：
- Shopify：Admin API 上传商品图
- WordPress：REST 媒体上传 + WooCommerce 关联
- 其他平台：仅打标

### 嵌入

```html
<iframe src="/embed.html?store=STORE_ID" style="width:100%;min-height:720px;border:0"></iframe>
```

## SDXL 引擎环境变量（v0.1.4）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VTON_AR` | `1` | 保宽高比 letterbox |
| `VTON_WARP` | `1` | 几何对齐 |
| `VTON_DETAIL` | `1` | 商品细节回贴（边缘衰减） |
| `VTON_LOCK_ID` | `1` | 身份锁（SeamlessClone） |
| `VTON_LORA` | `0.7` | LoRA scale |
| `VTON_IPA` | `0.9` | IP-Adapter 参考强度 |
| `VTON_CANNY` | 按品类 0.55/0.45 | Canny 边缘约束（实心面料 0.55 锁图案） |
| `VTON_GUIDANCE` | `7.5` | CFG guidance scale |
| `VTON_SEED` | `42` | 随机种子 |
| `VTON_PROMPT_HINT` | 空 | 额外 prompt 钩子 |

## 测试

```bash
# 后端（203 passed / 1 skipped）
cd backend
python3 -m pytest tests/ -q

# 前端（14 passed）
npm test
npm run build

# 静态检查
ruff check backend/ training/
ruff format --check backend/ training/
```

覆盖：单元（引擎/人种/CRUD/JWT/推送/worker/LoRA/并发/训练几何）、平台适配（clothing_mask/blend_tryon/reapply/detect_garment_extent）、组件（i18n/render）、集成（全 API 链路）、系统（真实 worker 出图）

## 目录结构

```text
backend/
  engine.py               合成引擎（CPU 兜底）
  engine_sdxl.py           SDXL 商用引擎（几何对齐 + ControlNet + IP-Adapter + 动态遮罩 + 细节回贴 + SeamlessClone）
  platform_presets.py      平台店规 + detect_garment_extent + clothing_mask + blend_tryon + reapply_garment_detail
  export_csv.py            CSV/SKU 批量导出
  lora_io.py               LoRA 加载（PEFT 优先）
  tests/                   pytest（203 passed）
src/                       前端 SPA + i18n
training/                  SDXL LoRA / ControlNet 训练脚本
models/sdxl/lora/          LoRA 权重（120 张训练，rank 16）
```

## 环境变量

```bash
STORE_JWT_SECRET=<至少 32 字节>
STORE_ADMIN_PASSWORD=<管理员密码>
STORE_STAFF_PASSWORD=<员工密码>
LICENSE_MODE=commercial
# VTON_IPA=0.9  VTON_CANNY=0.55  VTON_DETAIL=1  VTON_LOCK_ID=1 ...
```

## 引擎选择

| LICENSE_MODE | 引擎优先级 | 商用合规 |
|---|---|---|
| `commercial` | SDXL -> 合成引擎 | 合规 |
| `research` | IDM-VTON -> 合成引擎 | 不合规 |
| 未设置 | 合成引擎 | 需确认 |

## API 概览

```text
POST /api/auth/login
GET  /api/me  /api/stats  /api/settings
GET|POST /api/bases  /api/ethnicities  /api/stores
POST /api/jobs/batch  /api/jobs/export-csv
POST /api/stores/{id}/batch-push
POST /api/jobs/{id}/push                    # [admin]
GET  /api/jobs  /api/jobs/{id}  /api/download/{id}
GET|POST /api/uploads  /api/uploads/categories
GET  /api/health  /api/license
```

## 授权说明

- **IDM-VTON**：CC BY-NC-SA 4.0，禁止商用
- **SDXL + ControlNet + IP-Adapter（店铺换衣管线）**：商用权归作者所有
- **Atelier 核心代码**：MIT License

## 感谢

- [IDM-VTON](https://github.com/yisol/IDM-VTON) - 虚拟试衣模型（ECCV 2024）
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [Rembg](https://github.com/danielgatis/rembg)
