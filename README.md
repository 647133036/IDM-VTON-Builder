# Atelier v0.1.7

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

**基本运行条件：8 核 CPU / 32GB 内存 / NVIDIA A10（24GB 显存）。** SDXL 商用管线（ControlNet + IP-Adapter + LoRA）按此规格验证；无 GPU 时自动降级合成引擎，效果有限。

## v0.1.7 更新日志

### warp 垂直锚点对齐：修正 logo/花纹垂直偏位

v0.1.6 的 `_warp_garment` 仅做**水平**网格变形（沿躯干轮廓收窄/展宽），垂直方向是恒等映射。`fit_garment_to_body` 按肩宽等比缩放后，衣料的领口和下摆位置由初始放置决定，与人体颈部/腰部没有对齐——这是 pattern_lock 残余风险的主要来源：logo 偏高或偏低，花纹和褶皱不重合。

v0.1.7 在 `_warp_garment` 中加入**垂直锚点对齐**：

1. 从 alpha 通道检测衣料内容范围（领口行 → 下摆行）
2. 计算躯干框的垂直目标（颈部 `box["y"]` → 腰部 `box["y"]+box["h"]`）
3. 用 `np.interp` 构建非线性 `map_y`，将领口映射到颈部、下摆映射到腰部
4. 与水平 `map_x` 合成到同一次 `cv2.remap`，纯 CPU，无额外开销

```
VTON_WARP_VERTICAL=1.0  # 默认：全强度垂直对齐
VTON_WARP_VERTICAL=0.0  # 关闭：仅水平变形（v0.1.6 行为）
VTON_WARP_VERTICAL=0.5  # 半强度：在恒等和全对齐之间插值
```

**效果**：cloth_ref 的领口对齐颈部、下摆对齐腰部，pattern_lock overlay 的 logo 位置与人体姿态一致，不再因垂直偏位导致花纹错位。

### 残余风险

垂直对齐基于躯干框（`torso_box` 的人群/人种统计值），不是关键点检测。极端体型（超长躯干/超短腰）仍会偏；学习型 warp（TPS/光流）留待后续。

### GARMENT_BOX 适配：dress/lower/onesie 垂直范围修正

`align_garment_to_pose` 此前未应用 `GARMENT_BOX` 调整，连衣裙/裤子/连体衣的垂直 warp 目标与上衣相同（颈部→腰部），导致 dress 被压缩到躯干、lower 被贴到胸口。v0.1.7 在 warp 前按品类调整 `box["y"]` 和 `box["h"]`：

- `upper`（y=0, h=1.0）：颈部→腰部（无变化）
- `dress`（y=0, h=1.55）：颈部→膝部
- `lower`（y=0.92, h=0.95）：髋部→大腿
- `onesie`（y=0, h=1.35）：颈部→大腿

### _detect_skin 形态学去噪

YCbCr 肤色检测加入 5x5 椭圆核开运算（去散点）+ 闭运算（填小孔），避免散点噪声影响 `_correct_skin_tone` 的肤色均值统计和 `reapply_garment_detail` 的肤色排除。

### _correct_skin_tone 脸区排除

此前 `_correct_skin_tone` 对 result 中**所有**肤色像素做全局平移（`shift = orig_mean - res_mean`），包括 SDXL 通常生成正确的脸部。全局平移会让脸色不自然偏移。

v0.1.7 用前景 bbox 顶部 18% 估算脸区，从统计和应用两端排除：shift 只基于手臂/颈部肤色计算，校正只作用于手臂/颈部，脸部不动。

## v0.1.6 更新日志

### 核心突破：频率分离——让立体感和图案保真同时成立

虚拟换衣同时要两样东西：衣服穿在身上的褶皱光影要自然（立体感），logo/印花/条纹要和原衣一模一样（图案保真）。SDXL 在 mask 区重画衣服时，褶皱和图案被同一个生成过程绑定——光影对了图案就漂一笔，图案锁死了光影就变成贴纸。

PLTON（arXiv:2401.11239）提出了频率分离的解法：把服装特征拆成低频和高频，在去噪过程中分别处理。本管线借鉴 PLTON 论文的这一**频率分离**思想，将其集成到 SDXL 管线的 CPU 后处理层，提升了 SDXL 的图案保真能力：

- 把 warp 后的 `cloth_ref` 拆成**低频 LF**（底色、大块明暗）和**高频 HF**（花纹、logo、条纹边缘）
- 生成图的褶皱光影只调**低频**（底色全强度响应），**高频**基本不动（花纹半强度响应）
- 效果：阴影能压暗底色，但 logo 对比度只被半强度调制，不会随底色一起塌掉——**立体感和图案保真通过频率通道分离，各自独立处理**

```
lighting  = clamp(L_gen / L_ref, LIGHTING_MIN, LIGHTING_MAX)
shade_lf  = 1 + (lighting - 1) * light_strength * lf_strength   # 底色全强度
shade_hf  = 1 + (lighting - 1) * light_strength * hf_strength   # 花纹半强度(默认0.5)
lit       = LF_ref * shade_lf + HF_ref * shade_hf
```

lighting=1 时两个 shade 均为 1（无变化）；clamp 只作用于 lighting，shade_hf 半强度后自然落在 0.875–1.125，不再额外裁剪。`LIGHT_STRENGTH=0` 时 `lit = cloth_ref`（纯贴图回退基准）。

**管线本质**：SDXL 先生成一张「穿着目标款式、光影正确、但图案可能漂移」的图；后处理把这张图的光照信息提取出来，去调制原衣的图案信息，合成出「光影是 SDXL 的、图案是原衣的」最终结果。不是把贴图盖上去，是把两者的频率分量拆开重组。

### 10 个环境变量全接入

| # | 变量 | 默认 | 作用点 | 说明 |
|---|------|------|--------|------|
| 1 | `LF_BLUR_SIGMA` | 8.0 | 频率分离 | 高斯 σ，拆 LF/HF |
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

### light_strength 提升为 env var

旧：`_pattern_lock_overlay(..., light_strength=1.0, ...)` 函数参数。

新：`VTON_PATTERN_LOCK_LIGHT_STRENGTH` 环境变量，默认 1.0，在函数内部读取。调用方无需再传 `light_strength`；需要回退纯贴图时设 `VTON_PATTERN_LOCK_LIGHT_STRENGTH=0` 即可，不必改代码。

### LIGHT_STRENGTH=0 回退基准

lighting 仍按公式计算并 clamp，但 `shade_lf = shade_hf = 1`，因此 `lit = LF_ref + HF_ref = cloth_ref`。关闭光照调制，输出退化为纯 cloth_ref 贴图，保留白底跳过、肤色降权、mask 羽化三项空间处理。用于调试与回退。

### 白底跳过可控

- `SKIP_WHITE_BG=0`：不跳过任何白底，valid 全为 True。
- `SKIP_WHITE_BG=1` + `WHITE_THRESH`：只跳过 RGB 均大于阈值、且与画布边缘连通的白色区域；白衣内部不连通，仍会被覆盖。
- 阈值调高（如 250）→ 只跳极白背景；调低（如 235）→ 更激进地跳白，可能误伤浅色衣物边缘，需配合 `FEATHER_PX` 观察接缝。

### 调参示例

```bash
# 默认：底色全强度、花纹半强度
export VTON_PATTERN_LOCK=1
export VTON_PATTERN_LOCK_LIGHT_STRENGTH=1.0
export VTON_PATTERN_LOCK_LF_STRENGTH=1.0
export VTON_PATTERN_LOCK_HF_STRENGTH=0.5

# 强对比 logo：花纹几乎不随光照变暗
export VTON_PATTERN_LOCK_HF_STRENGTH=0.2

# 关闭光照调制（精确 ref，仅贴图覆盖）
export VTON_PATTERN_LOCK_LIGHT_STRENGTH=0.0

# 细条纹：低频 σ 调小，避免条纹被算进低频
export VTON_PATTERN_LOCK_LF_BLUR_SIGMA=5.0
export VTON_PATTERN_LOCK_HF_STRENGTH=0.4
```

### 残余风险

频率分离后处理借 PLTON 的 LF/HF 解耦思想，不换模型：底色全强度调光保立体感，花纹半强度调光保 logo 对比度。残余风险为 warp 对齐误差和接缝处 HF 不连续，与纯贴图管线相同。

## v0.1.5 更新日志（历史）

本轮把换衣后遗症当图像修复问题处理：胸口白块（领口接缝融合失败）、手臂变白（短袖 mask 把上臂当生成区、SDXL 重画丢肤色）、上衣沿裤腰齐边剪（mask 固定百分比截断）、图案被 SDXL 重画。CPU 层立刻生效；SDXL Inpainting 二次修复按需开启。

### 修改了什么

| 缺陷 | 根因 | 本轮改动 |
|------|------|----------|
| **胸口白块** | `force_white_background` 未传主体 mask；cv2 5.0 人脸检测全零，脸/脖子豁免失效；领口接缝硬拼 | 主体 mask 算一次，同时传给 `clothing_mask` 和 `force_white_background`；heuristic 脸区（主体 bbox 顶 18%）兜底；`_seam_repair` 对 mask 边界 8px 带做 Telea inpaint |
| **手臂变白** | 短袖 mask=255 含上臂，`reapply` 把衣料纹理贴到皮肤，SDXL 重画肤色 | 短袖/无袖从主体外侧 25% 减去 YCbCr 肤色（原图手臂保留）；`reapply` 对肤色区降 85% 纹理/颜色；`_correct_skin_tone` 把生成肤色拉向原图 |
| **上衣齐裤腰裁** | 上衣 mask 按主体高度 62% 横切，裤腰以下贴回原图 | 已看 `hem`：长款 80% 垂过裤腰；短袖齐腰仍 62%。OpenPose 腰点仍未用于裁切 |
| **图案被改** | SDXL 在 mask=255 区重画衣服，Canny/IP-Adapter 只是引导，logo/印花/条纹总会漂一笔 | DualFit 贴图：`VTON_PATTERN_LOCK=1` 用 warp 后的 `cloth_ref` 覆盖衣服区，PLTON 频率分离（底色全强度调光、花纹半强度调），生成图只调褶皱光影。**残余风险**：①warp 对齐误差导致 logo 位置偏；②接缝羽化区高频不连续，细条纹在袖口/领口可能穿帮 |

### 增加了什么

- **`_heuristic_face_mask`**：cv2 5.0 去掉 `CascadeClassifier` 后，用主体 bbox 顶 18% 估脸区，`clothing_mask` 与 `force_white_background` 共用。
- **`_detect_skin` / `_correct_skin_tone`**：YCbCr 肤色（Cb 77–127，Cr 133–173）。注意 OpenCV `COLOR_RGB2YCrCb` 通道顺序是 `[Y, Cr, Cb]`。
- **`_seam_repair`**：`blend_tryon` 之后对 mask 边界 dilate-erode 带做 `cv2.INPAINT_TELEA`，用周边像素填接缝。
- **`_sdxl_inpaint_repair`**：`VTON_INPAINT_REPAIR=1` 且 `models/sdxl/sdxl_inpaint/` 有权重时，用官方 inpaint 模型低降噪（0.35）局部重绘。无权重时跳过。
- **短袖手臂保护**：`clothing_mask` 仅在 `upper`+短袖/无袖时从主体水平外侧 25% 抠肤色；长袖不抠，袖子仍可盖手臂。
- **图案锁 `_pattern_lock_overlay`**：`VTON_PATTERN_LOCK=1`（默认，且 `VTON_WARP=1`）时，warp 后的 `cloth_ref` 覆盖衣服区。借鉴 PLTON（arXiv:2401.11239）的 HF-Map 思路做频率分离：底色按 `LF_STRENGTH`（1.0）全强度调褶皱光影、花纹按 `HF_STRENGTH`（0.5）半强度调——阴影能压暗底色但 logo 对比度不塌。白底只跳过与画布边缘连通的近白；肤色区降权；mask 边缘羽化。`VTON_PATTERN_LOCK=0` 回退高频柔光 + Reinhard。**残余风险**：warp 对齐误差致 logo 偏位；接缝羽化区高频不连续。PLTON 那种去噪过程 U-Net 注入需要换模型，本管线做不到。（v0.1.6 定稿：10 env var 全接入，见 v0.1.6 更新日志）

### 完善了什么

- 管线顺序：`reapply`（pattern_lock 或柔光）→ `blend_tryon` → `_seam_repair` → `_correct_skin_tone` → `_sdxl_inpaint_repair` → `force_white_background(subject_mask)`。
- 品类：前端选大类（`auto` / 上衣 / 裙装 / 下装 / 连体）；短袖/长袖/无袖、齐腰/过腰由后端 `detect_garment_extent` 从平铺图几何推断。`auto`：aspect>1.6→连衣裙，<0.7→裤子，其余→上衣。
- 单测：244 passed / 1 skipped（新增肤色检测、heuristic 脸区、接缝 inpaint、短袖排除手臂、Inpainting 骨架、pattern_lock 贴图 + 频率分离）。

## v0.1.4 更新日志（历史）

v0.1.4 解决换图案、衣袖截断、胸口深 V / 肩膀重影，并引入动态遮罩与 SeamlessClone。Canny 按品类 0.55/0.45，IP-Adapter 0.9，texture_strength 0.70；`detect_garment_extent` + 按 hem/sleeve 裁 mask；`blend_tryon` 连通分量 SeamlessClone。

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
- **动态 `clothing_mask`（商业级遮罩）**：上衣同时看衣长(`hem`)和袖长(`sleeve`)决定裁切范围：
  - 长款上衣(hem=long) → 80% 身高（垂过裤腰，解决「沿裤子齐边剪」问题）
  - 长袖常规(hem=waist) → 72%（含手臂区，替代原 88% 过度延伸躯干）
  - 短袖常规 → 62%
  - 无袖 → 55%（不含手臂）
  - 连衣裙 → 按衣长决定（过腰 96% / 正常 82%）
- **`auto` 类型自动推断**：`garment_type=auto`（默认）时从平铺衣料宽高比推断——aspect>1.6→dress，<0.7→lower，其他→upper。不再穿透无裁切。
- **`blend_tryon` 升级为 SeamlessClone**：身份区（mask=0，脸/手/脚）按连通分量逐块用 `cv2.seamlessClone(NORMAL_CLONE)` 无缝贴回原图；质心落在孔洞外时自动取组件内中位点；衣服区用增强 alpha 混合（×3 权重）锁死生成结果；seamlessClone 失败时安全回退羽化混合。

### 实现了什么功能

商业级换衣流程三步落地：

```
1. 原图剥离    detect_garment_extent → 分析衣料袖长/衣长
                clothing_mask        → 动态遮罩:长袖含手臂,无袖不含

2. AI 重绘     SDXL + ControlNet(Canny 0.55) + IP-Adapter(0.9)
                reapply_garment_detail
                  VTON_PATTERN_LOCK=1 → warp 衣覆盖衣服区,生成图只调光影
                  VTON_PATTERN_LOCK=0 → 高频柔光 + Reinhard 颜色迁移

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
- **v0.1.6**：pattern_lock 频率分离定稿（10 env var 全接入，light_strength 提升为环境变量，LIGHT_STRENGTH=0 纯贴图回退）
- **v0.1.5**：胸口白块 / 手臂变白修复（主体 mask 压白保护、短袖排除手臂肤色、Telea 接缝、肤色校正）；可选 SDXL Inpainting 二次修复
- **v0.1.4**：动态袖长检测 + 动态遮罩 + SeamlessClone + Canny/IP-Adapter/texture 权重提升

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

**生成处理（v0.1.5 商业级流程）**：
1. `detect_garment_extent` 分析衣料袖长/衣长；`auto` 按 aspect 推断品类
2. `crop_garment_to_content` 裁剪主体
3. `align_garment_to_pose` 几何对齐送 IP-Adapter
4. ControlNet（OpenPose 0.8 + Canny 0.55 锁图案 + Depth 0.95）+ IP-Adapter 0.9
5. SDXL 生成姿态/光影/人体
6. `reapply_garment_detail` 边缘衰减贴回纹理；肤色区降 85% 强度
7. `blend_tryon` 动态 mask + SeamlessClone 贴回脸/手/脚（短袖手臂为 mask=0）
8. `_seam_repair` Telea 填 mask 边界接缝
9. `_correct_skin_tone` 生成肤色拉向原图
10. `_sdxl_inpaint_repair`（可选）低降噪局部重绘
11. `force_white_background(subject_mask)` 纯白底，主体/胸口不压白

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

实测版本：PyTorch 2.13.0 + cu130 / diffusers 0.40.0（A10 23GB 验证）

## 多店铺与推送

### CSV/SKU 批量推送

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

## SDXL 引擎环境变量（v0.1.7）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VTON_AR` | `1` | 保宽高比 letterbox |
| `VTON_WARP` | `1` | 几何对齐 |
| `VTON_WARP_VERTICAL` | `1.0` | 垂直锚点对齐强度：1.0=领口→颈部/下摆→腰部全对齐，0=仅水平变形（v0.1.6 行为） |
| `VTON_DETAIL` | `1` | 商品细节回贴（边缘衰减；肤色区降权） |
| `VTON_PATTERN_LOCK` | `1` | DualFit 贴图：warp 衣覆盖衣服区，生成图只按亮度比调褶皱；白底跳过边缘连通近白；肤色降权。需 `VTON_WARP=1`。`0` 回退高频柔光+Reinhard |
| `VTON_PATTERN_LOCK_LIGHT_STRENGTH` | `1.0` | 光照总闸：0=精确 ref（纯贴图），1=全效 |
| `VTON_PATTERN_LOCK_LF_STRENGTH` | `1.0` | 底色光照响应系数（1=全强度），褶皱阴影全量压暗底色 |
| `VTON_PATTERN_LOCK_HF_STRENGTH` | `0.5` | 花纹光照响应系数（0=花纹不受光照，1=全强度）。深色底+亮色 logo 用 0.3 |
| `VTON_PATTERN_LOCK_LF_BLUR_SIGMA` | `8.0` | 低频提取高斯 σ。细条纹 5–7；大块印花 9–13；logo 文字 11–15 |
| `VTON_PATTERN_LOCK_LIGHTING_MIN` | `0.75` | 亮度比下限（clamp 作用于 lighting，非 shade） |
| `VTON_PATTERN_LOCK_LIGHTING_MAX` | `1.25` | 亮度比上限（clamp 作用于 lighting，非 shade） |
| `VTON_PATTERN_LOCK_SKIN_WEIGHT` | `0.85` | 肤色区降权（mask 内手臂皮肤不被贴服装色） |
| `VTON_PATTERN_LOCK_FEATHER_PX` | `5` | mask 边缘羽化像素（接缝过渡宽度） |
| `VTON_PATTERN_LOCK_SKIP_WHITE_BG` | `1` | 跳过与画布边缘连通的白底；`0` 关闭（覆盖所有像素含白底） |
| `VTON_PATTERN_LOCK_WHITE_THRESH` | `245` | 白底 RGB 判定阈值（三通道均大于此值视为近白） |
| `VTON_LOCK_ID` | `1` | 身份锁（SeamlessClone） |
| `VTON_LORA` | `0.7` | LoRA scale |
| `VTON_IPA` | `0.9` | IP-Adapter 参考强度 |
| `VTON_CANNY` | 按品类 0.55/0.45 | Canny 边缘约束（实心面料 0.55 锁图案） |
| `VTON_GUIDANCE` | `7.5` | CFG guidance scale |
| `VTON_SEED` | `42` | 随机种子 |
| `VTON_PROMPT_HINT` | 空 | 额外 prompt 钩子 |
| `VTON_INPAINT_REPAIR` | `0` | 设为 `1` 且已放置 `models/sdxl/sdxl_inpaint/` 时启用 SDXL Inpainting 二次修复 |

## 测试

```bash
# 后端（244 passed / 1 skipped）
cd backend
python3 -m pytest tests/ -q

# 前端（14 passed）
npm test
npm run build

# 静态检查
ruff check backend/ training/
ruff format --check backend/ training/
```

覆盖：单元（引擎/人种/CRUD/JWT/推送/worker/LoRA/并发/训练几何）、平台适配（clothing_mask/blend_tryon/reapply/detect_garment_extent/肤色/接缝 inpaint）、组件（i18n/render）、集成（全 API 链路）、系统（真实 worker 出图）

## 目录结构

```text
backend/
  engine.py               合成引擎（CPU 兜底）
  engine_sdxl.py           SDXL 商用引擎（对齐 + ControlNet + IP-Adapter + 动态遮罩 + 图案锁 + 接缝/肤色后处理）
  platform_presets.py      店规 + detect_garment_extent + clothing_mask + blend_tryon + reapply / pattern_lock + seam/skin
  export_csv.py            CSV/SKU 批量导出
  lora_io.py               LoRA 加载（PEFT 优先）
  tests/                   pytest（244 passed）
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
# VTON_IPA=0.9  VTON_CANNY=0.55  VTON_DETAIL=1  VTON_PATTERN_LOCK=1  VTON_PATTERN_LOCK_HF_STRENGTH=0.5  VTON_PATTERN_LOCK_LIGHT_STRENGTH=1.0  VTON_LOCK_ID=1 ...
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
