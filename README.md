# Atelier v0.1.8

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

**基本运行条件：8 核 CPU / 32GB 内存 / NVIDIA A10（24GB 显存）。** SDXL 商用管线（ControlNet + IP-Adapter + LoRA）按此规格验证；无 GPU 时自动降级合成引擎，效果有限。

## v0.1.8 更新日志

### RTMPose 关键点检测：替代 MediaPipe，精度提升 + 侧背面退化更小

v0.1.7 用 MediaPipe Pose 检测髋部关键点辅助腰线估算。实测 hip_y_ratio=0.681，比轮廓法回退值 0.62 高出 6 个百分点，且 MediaPipe 在侧面/背面姿态下精度退化严重（AP 下降 13.7%）。

v0.1.8 切换为 RTMPose-m（COCO AP 75.8%，CPU 90+ FPS，侧/背面退化仅 2.2%）：

- **模型**：RTMPose-m（CSPNeXt backbone + RTMCCHead，SimCC 输出，输入 256x192）
- **推理**：从灰度阈值提取 bbox → expand 1.25 → resize → ImageNet normalize → backbone → head → SimCC argmax 解码
- **依赖绕过**：fake `xtcocotools`/`mmdet`/`mmcv.ops` 注入 `sys.modules`，绕过 mmpose import 链中对编译扩展和数据集/评估模块的依赖。推理只需要 backbone + head，不需要这些模块。`RTMPOSE_FAKE_DEPS=1`（默认）开启，设 `0` 关闭（真实 mmcv/mmdet 已装时）
- **PyTorch 2.6 兼容**：`torch.load(weights_only=False)` 加载含 numpy 数组的 checkpoint
- **MediaPipe 回退**：RTMPose 初始化或推理失败时自动降级到 MediaPipe Pose
- **SimCC 解码**：`_simcc_decode` 独立纯函数，crop 正方形 bbox 直接 resize，x/y 独立缩放，无 letterbox padding；归一化分母为原图宽高

### 腰线插值修正：COCO hip landmark 不等于服装腰线

切换到 RTMPose 后发现更深层的概念问题：COCO 标注的 "hip" landmark 在大转子（股骨）位置，比服装腰线低 10-15 个百分点。直接用 hip_y 作为 waist_r 会导致上衣下摆过长、裤腰位置偏低。

v0.1.8 改用 shoulder → hip 插值：

```
waist_y = shoulder_y + (hip_y - shoulder_y) * 0.55
```

腰线定位在肩部和髋部之间偏上 55% 处，对应人体自然腰线（肚脐水平）。

| 方法 | hip_y_ratio（直接） | waist_r（插值后） |
|------|---------------------|-------------------|
| 轮廓法回退 | — | 0.62 |
| MediaPipe | 0.681 | — |
| RTMPose | 0.725 | **0.544** |

插值后的 0.544 处于人体腰线的正确范围（0.50-0.58），三条路径（pose/轮廓/默认）的输出一致区间收窄。

### Dress mask 修复：脖子覆盖 + 腿部裁切

v0.1.7 的 dress/onesie 遮罩有两个问题：

1. **脖子残留**：face 排除只覆盖顶部 18%，脖子区（18-22%）残留在 mask 中，导致脖子被贴上服装纹理
2. **腿部未覆盖**：dress normal 下摆在 `waist_r + 0.20`（约 0.82），只到大腿中段，膝盖以下露出原图

v0.1.8 修复：

- **脖子排除**：dress/onesie 类型在 face 排除后额外清零顶部 22%（face 18% + neck 4%）。upper/lower 领口在衣物范围内，不排除
- **腿部覆盖**：dress normal 下摆从 `waist_r + 0.20` 改为 `waist_r + 0.28`（约 0.90，膝部）；long 从 `+0.34` 改为 `+0.35`。onesie normal 同步改为 `+0.28`
- **knee 关键点精确截断**：有 knee 关键点时，dress/onesie 下摆截到 `knee_r - 0.02`（normal）或 `knee_r + 0.05`（long），无关键点回退固定 offset

### 残余风险

RTMPose 依赖 fake 模块注入绕过 mmpose import 链（`RTMPOSE_FAKE_DEPS=1`）。mmpose 升级可能新增 import 路径导致 fake 失效，此时静默回退 MediaPipe。安装完整 mmcv 后设 `RTMPOSE_FAKE_DEPS=0`。SimCC `split_ratio=2.0` 硬编码，换模型需同步。shoulder/hip 任一 vis < 0.3 回退轮廓法（RTMPose 路径 vis 固定 1.0，阈值只约束 MediaPipe）。

## v0.1.7 更新日志（历史）

### warp 垂直锚点对齐：修正 logo/花纹垂直偏位

v0.1.6 的 `_warp_garment` 仅做水平网格变形（沿躯干轮廓收窄/展宽），垂直方向是恒等映射。`fit_garment_to_body` 按肩宽等比缩放后，衣料的领口和下摆位置由初始放置决定，与人体颈部/腰部没有对齐——这是 pattern_lock 残余风险的主要来源：logo 偏高或偏低，花纹和褶皱不重合。

v0.1.7 在 `_warp_garment` 中加入垂直锚点对齐：

1. 从 alpha 通道检测衣料内容范围（领口行 → 下摆行）
2. 计算躯干框的垂直目标（颈部 `box["y"]` → 腰部 `box["y"]+box["h"]`）
3. 用 `np.interp` 构建非线性 `map_y`，将领口映射到颈部、下摆映射到腰部
4. 与水平 `map_x` 合成到同一次 `cv2.remap`，纯 CPU，无额外开销

```
VTON_WARP_VERTICAL=1.0  # 默认：全强度垂直对齐
VTON_WARP_VERTICAL=0.0  # 关闭：仅水平变形（v0.1.6 行为）
VTON_WARP_VERTICAL=0.5  # 半强度：在恒等和全对齐之间插值
```

### GARMENT_BOX 适配：dress/lower/onesie 垂直范围修正

`align_garment_to_pose` 此前未应用 `GARMENT_BOX` 调整，连衣裙/裤子/连体衣的垂直 warp 目标与上衣相同（颈部→腰部），导致 dress 被压缩到躯干、lower 被贴到胸口。v0.1.7 在 warp 前按品类调整 `box["y"]` 和 `box["h"]`：

- `upper`（y=0, h=1.0）：颈部→腰部（无变化）
- `dress`（y=0, h=1.55）：颈部→膝部
- `lower`（y=0.92, h=0.95）：髋部→大腿
- `onesie`（y=0, h=1.35）：颈部→大腿

### _detect_skin 形态学去噪

YCbCr 肤色检测加入 5x5 椭圆核开运算（去散点）+ 闭运算（填小孔），避免散点噪声影响 `_correct_skin_tone` 的肤色均值统计和 `reapply_garment_detail` 的肤色排除。

### _correct_skin_tone 脸区排除

此前 `_correct_skin_tone` 对 result 中所有肤色像素做全局平移（`shift = orig_mean - res_mean`），包括 SDXL 通常生成正确的脸部。全局平移会让脸色不自然偏移。

v0.1.7 用前景 bbox 顶部 18% 估算脸区，从统计和应用两端排除：shift 只基于手臂/颈部肤色计算，校正只作用于手臂/颈部，脸部不动。

### _sdxl_inpaint_repair 可用化

v0.1.6 的 `_sdxl_inpaint_repair` 有三个问题：遮罩是整张 shared_mask（inpaint 重画整个衣服区）、每次调用 from_pretrained + del pipe（OOM 风险高）、无原图对比（无法定位肤色异常区）。

v0.1.7 修复：聚焦遮罩（接缝带 + 肤色异常区）、管线单例（`_INPAINT_PIPE` + `_INPAINT_LOCK`）、original 参数传入。启用条件：`VTON_INPAINT_REPAIR=1` + `models/sdxl/sdxl_inpaint/model_index.json`。

### 腰线启发式检测

`clothing_mask` 此前用固定比例裁切衣料 mask 底部，v0.1.7 加入 `_estimate_waist_ratio`：从 subject_mask 行宽分布检测躯干区最窄点作为腰线。v0.1.8 进一步加入 RTMPose 关键点插值（见上方 v0.1.8 更新日志）。

| 品类 | 条件 | 原比例 | 新公式 |
|------|------|--------|--------|
| upper | short | 0.62 | waist_r |
| upper | long sleeve | 0.72 | waist_r + 0.10 |
| upper | sleeveless | 0.55 | waist_r - 0.07 |
| upper | long hem | 0.80 | waist_r + 0.18 |
| dress | normal | 0.82 | waist_r + 0.28 |
| dress | long | 0.96 | waist_r + 0.35 |
| lower | top cut | 0.48 | waist_r - 0.14 |
| onesie | normal | 0.82 | waist_r + 0.28 |
| onesie | long sleeve | 0.95 | waist_r + 0.33 |

## v0.1.6 更新日志（历史）

### 核心突破：频率分离——让立体感和图案保真同时成立

虚拟换衣同时要两样东西：衣服穿在身上的褶皱光影要自然（立体感），logo/印花/条纹要和原衣一模一样（图案保真）。SDXL 在 mask 区重画衣服时，褶皱和图案被同一个生成过程绑定——光影对了图案就漂一笔，图案锁死了光影就变成贴纸。

PLTON（arXiv:2401.11239）提出了频率分离的解法：把服装特征拆成低频和高频，在去噪过程中分别处理。本管线借鉴 PLTON 论文的频率分离思想，将其集成到 SDXL 管线的 CPU 后处理层：

- 把 warp 后的 `cloth_ref` 拆成低频 LF（底色、大块明暗）和高频 HF（花纹、logo、条纹边缘）
- 生成图的褶皱光影只调低频（底色全强度响应），高频基本不动（花纹半强度响应）
- 效果：阴影能压暗底色，但 logo 对比度只被半强度调制，不会随底色一起塌掉

```
lighting  = clamp(L_gen / L_ref, LIGHTING_MIN, LIGHTING_MAX)
shade_lf  = 1 + (lighting - 1) * light_strength * lf_strength
shade_hf  = 1 + (lighting - 1) * light_strength * hf_strength
lit       = LF_ref * shade_lf + HF_ref * shade_hf
```

### 10 个环境变量全接入

| # | 变量 | 默认 | 作用点 | 说明 |
|---|------|------|--------|------|
| 1 | `LF_BLUR_SIGMA` | 8.0 | 频率分离 | 高斯 sigma，拆 LF/HF |
| 2 | `LIGHTING_MIN` | 0.75 | lighting clamp | 亮度比下限 |
| 3 | `LIGHTING_MAX` | 1.25 | lighting clamp | 亮度比上限 |
| 4 | `LIGHT_STRENGTH` | 1.0 | shade_lf/shade_hf | 光照总闸，0=纯贴图回退 |
| 5 | `LF_STRENGTH` | 1.0 | shade_lf | 底色光照响应 |
| 6 | `HF_STRENGTH` | 0.5 | shade_hf | 花纹光照响应 |
| 7 | `SKIN_WEIGHT` | 0.85 | alpha | 肤色区覆盖降权 |
| 8 | `FEATHER_PX` | 5 | alpha | mask 边缘羽化半径 |
| 9 | `SKIP_WHITE_BG` | 1 | valid | 白底跳过开关 |
| 10 | `WHITE_THRESH` | 245 | valid | 白底判定阈值 |

（均带 `VTON_PATTERN_LOCK_` 前缀）

## v0.1.5 更新日志（历史）

本轮把换衣后遗症当图像修复问题处理：胸口白块、手臂变白、上衣沿裤腰齐边剪、图案被 SDXL 重画。CPU 层立刻生效；SDXL Inpainting 二次修复按需开启。

| 缺陷 | 根因 | 改动 |
|------|------|------|
| 胸口白块 | force_white_background 未传主体 mask；cv2 5.0 人脸检测全零 | 主体 mask 算一次传给 clothing_mask 和 force_white_background；heuristic 脸区兜底；_seam_repair Telea inpaint |
| 手臂变白 | 短袖 mask 含上臂，reapply 贴衣料到皮肤 | 短袖/无袖从主体外侧 25% 减去 YCbCr 肤色；reapply 肤色区降 85%；_correct_skin_tone 拉回原图 |
| 上衣齐裤腰 | mask 按主体高度 62% 横切 | 按 hem 判定：长款 80% 垂过裤腰 |
| 图案被改 | SDXL 在 mask=255 区重画 | pattern_lock：warp 后 cloth_ref 覆盖衣服区，频率分离调光 |

## v0.1.4 更新日志（历史）

Canny 按品类 0.55/0.45，IP-Adapter 0.9，texture_strength 0.70；`detect_garment_extent` + 按 hem/sleeve 裁 mask；`blend_tryon` 连通分量 SeamlessClone；`face_margin` 非对称扩展修复胸口深 V；`reapply_garment_detail` 边缘衰减修复肩膀重影。

## v0.1.3 更新日志（历史）

引入「AI 出骨架光影，代码保商品细节」混合生成策略：`align_garment_to_pose` 几何对齐 + `reapply_garment_detail` 纹理颜色回贴 + `blend_tryon` 身份锁。修复 LoRA 权重加载（PEFT 优先）、训练几何（time_ids / letterbox / VAE fp32 / min-SNR）、推理默认值。

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

## 安全

本项目经过两轮安全审查。当前状态如下：

### 已修复

- 文件服务鉴权：`/files/{path}` 仅暴露 `uploads`/`results`/`bases` 白名单目录，仅允许图片后缀
- 上传路径校验：文件名和分类名通过正则白名单校验，拒绝路径穿越
- 图片内容验证：PIL 解码校验，仅接受 PNG/JPEG/WEBP，最大 20MB
- JWT 密钥：无硬编码默认值；未设置时随机生成并持久化（chmod 600）
- 账号密码：无公开默认密码；未设置时随机生成并输出到 stdout
- 角色权限：admin 可执行所有操作，staff 仅可创建任务和查看
- 凭据脱敏：`/api/stores` 响应不返回明文 api_key
- SSRF 防护：`/api/push` 强制 https + `.myshopify.com`，拒绝内网地址
- XSS 防护：前端 `escapeHtml` 转义；服务端枚举白名单校验
- GPU 并发：扩散推理路径全局 `threading.Lock` 串行化
- SDXL 管线缓存：构建一次后常驻
- 批量事务性：`/api/jobs/batch` 先全量校验再统一写盘

### 已知限制

- DNS 重绑定（TOCTOU）未完全消除
- 无 GPU 环境下合成引擎为启发式近似，效果有限
- 去水印为浅色角部 best-effort

## 图片规格

**上传**：PNG / JPEG / WEBP，最大 20MB，任意尺寸

**生成处理**：
1. `detect_garment_extent` 分析衣料袖长/衣长；`auto` 按 aspect 推断品类
2. `crop_garment_to_content` 裁剪主体
3. `align_garment_to_pose` 几何对齐送 IP-Adapter
4. ControlNet（OpenPose 0.8 + Canny 0.55 锁图案 + Depth 0.95）+ IP-Adapter 0.9
5. SDXL 生成姿态/光影/人体
6. `reapply_garment_detail` 边缘衰减贴回纹理；肤色区降 85% 强度
7. `blend_tryon` 动态 mask + SeamlessClone 贴回脸/手/脚
8. `_seam_repair` Telea 填 mask 边界接缝
9. `_correct_skin_tone` 生成肤色拉向原图（排除脸区）
10. `_sdxl_inpaint_repair`（可选）低降噪局部重绘
11. `force_white_background(subject_mask)` 纯白底

**输出**：长边 2048px，宽高比跟随底座图

## 技术栈

**前端**：原生 JavaScript（ES 模块）+ Vite 5 + 自研 i18n（zh/en）

**后端**：Python 3 + FastAPI + Uvicorn + PyJWT + PIL / NumPy / OpenCV + JSON 文件持久化

**AI 引擎**：
- IDM-VTON（CC BY-NC-SA 4.0，禁止商用）
- SDXL + ControlNet + IP-Adapter（商用合规）
- 合成引擎（无 GPU 兜底）

**姿态检测**：
- RTMPose-m（CSPNeXt + RTMCCHead，COCO 17 关键点，CPU 推理）
- MediaPipe Pose（回退）

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

## 运行条件

| 项目 | 规格 |
|------|------|
| CPU | 8 核 |
| 内存 | 32GB |
| GPU | NVIDIA A10（24GB 显存） |
| 系统 | Linux，CUDA 驱动与 PyTorch 匹配 |

A10 上 SDXL + 双 ControlNet + IP-Adapter 使用 `enable_model_cpu_offload`，VAE fp32 解码。低于此规格可跑合成引擎或关 Depth ControlNet。训练 LoRA 同样按 A10 23GB 验证（约 11 分钟 / 800 步）。

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

## 多店铺与推送

### CSV/SKU 批量推送

- 任务队列页导出 CSV+图：POST `/api/jobs/export-csv` 下载 ZIP
- 店铺卡片批量推送：上传 CSV/ZIP，POST `/api/stores/{id}/batch-push`
- 替换图优先于原成品，SKU 回退链：CSV 行 sku > 店铺 default_sku > 空

### 单张推送

完成的任务勾选店铺推送（admin 权限）：
- Shopify：Admin API 上传商品图
- WordPress：REST 媒体上传 + WooCommerce 关联
- 其他平台：仅打标

### 嵌入

```html
<iframe src="/embed.html?store=STORE_ID" style="width:100%;min-height:720px;border:0"></iframe>
```

## SDXL 引擎环境变量（v0.1.8）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VTON_AR` | `1` | 保宽高比 letterbox |
| `VTON_WARP` | `1` | 几何对齐 |
| `VTON_WARP_VERTICAL` | `1.0` | 垂直锚点对齐强度 |
| `VTON_DETAIL` | `1` | 商品细节回贴（边缘衰减；肤色区降权） |
| `VTON_PATTERN_LOCK` | `1` | DualFit 贴图 + 频率分离 |
| `VTON_PATTERN_LOCK_LIGHT_STRENGTH` | `1.0` | 光照总闸：0=纯贴图，1=全效 |
| `VTON_PATTERN_LOCK_LF_STRENGTH` | `1.0` | 底色光照响应系数 |
| `VTON_PATTERN_LOCK_HF_STRENGTH` | `0.5` | 花纹光照响应系数 |
| `VTON_PATTERN_LOCK_LF_BLUR_SIGMA` | `8.0` | 低频提取高斯 sigma |
| `VTON_PATTERN_LOCK_LIGHTING_MIN` | `0.75` | 亮度比下限 |
| `VTON_PATTERN_LOCK_LIGHTING_MAX` | `1.25` | 亮度比上限 |
| `VTON_PATTERN_LOCK_SKIN_WEIGHT` | `0.85` | 肤色区降权 |
| `VTON_PATTERN_LOCK_FEATHER_PX` | `5` | mask 边缘羽化像素 |
| `VTON_PATTERN_LOCK_SKIP_WHITE_BG` | `1` | 跳过边缘连通白底 |
| `VTON_PATTERN_LOCK_WHITE_THRESH` | `245` | 白底 RGB 判定阈值 |
| `VTON_LOCK_ID` | `1` | 身份锁（SeamlessClone） |
| `VTON_LORA` | `0.7` | LoRA scale |
| `VTON_IPA` | `0.9` | IP-Adapter 参考强度 |
| `VTON_CANNY` | 按品类 0.55/0.45 | Canny 边缘约束 |
| `VTON_GUIDANCE` | `7.5` | CFG guidance scale |
| `VTON_SEED` | `42` | 随机种子 |
| `VTON_PROMPT_HINT` | 空 | 额外 prompt 钩子 |
| `VTON_INPAINT_REPAIR` | `0` | 设为 `1` 且有权重时启用 SDXL Inpainting 二次修复 |
| `RTMPOSE_FAKE_DEPS` | `1` | 注入空壳 xtcocotools/mmdet/mmcv.ops；真实依赖已装时设 `0` |

## 测试

```bash
# 后端（265 passed / 1 skipped）
cd backend
python3 -m pytest tests/ -q

# 前端（14 passed）
npm test
npm run build

# 静态检查
ruff check backend/ training/
ruff format --check backend/ training/
```

覆盖：单元（引擎/人种/CRUD/JWT/推送/worker/LoRA/并发/训练几何）、平台适配（clothing_mask/blend_tryon/reapply/detect_garment_extent/肤色/接缝 inpaint/pose 关键点/dress 脖子排除/dress 腿部覆盖/knee 截断）、组件（i18n/render）、集成（全 API 链路）、系统（真实 worker 出图）

## 目录结构

```text
backend/
  engine.py               合成引擎（CPU 兜底）
  engine_sdxl.py           SDXL 商用引擎（对齐 + ControlNet + IP-Adapter + 动态遮罩 + 图案锁 + 接缝/肤色后处理）
  platform_presets.py      店规 + detect_garment_extent + clothing_mask + blend_tryon + reapply / pattern_lock + seam/skin + RTMPose/MediaPipe 关键点检测
  export_csv.py            CSV/SKU 批量导出
  lora_io.py               LoRA 加载（PEFT 优先）
  tests/                   pytest（265 passed）
src/                       前端 SPA + i18n
training/                  SDXL LoRA / ControlNet 训练脚本
models/sdxl/lora/          LoRA 权重（120 张训练，rank 16）
models/rtmpose/            RTMPose-m checkpoint（需手动下载）
```

## 环境变量

```bash
STORE_JWT_SECRET=<至少 32 字节>
STORE_ADMIN_PASSWORD=<管理员密码>
STORE_STAFF_PASSWORD=<员工密码>
LICENSE_MODE=commercial
# VTON_IPA=0.9  VTON_CANNY=0.55  VTON_DETAIL=1  VTON_PATTERN_LOCK=1 ...
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
- **RTMPose**：Apache 2.0，商用合规
- **Atelier 核心代码**：MIT License

## 感谢

- [IDM-VTON](https://github.com/yisol/IDM-VTON) - 虚拟试衣模型（ECCV 2024）
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - 实时关键点检测
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [Rembg](https://github.com/danielgatis/rembg)
