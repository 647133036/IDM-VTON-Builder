# 许可证合规说明

## 核心问题

IDM-VTON 采用 **CC BY-NC-SA 4.0** 许可，其中 **NC** 条款禁止商业用途。

电商换衣是商业用途，与 IDM-VTON 许可直接冲突。

## 合规路径

### 路径 A：研究/测试用途（合规）

- 内部测试、学习、学术验证
- 不面向真实用户，不产生商业收益
- 必须署名（BY）

```bash
LICENSE_MODE=research
```

### 路径 B：商业用途（需替代方案）

IDM-VTON 不可用于生产环境。需要替换为允许商用的模型：

| 组件 | 许可 | 商用状态 |
|------|------|----------|
| Stable Diffusion XL | OpenRAIL-M / Stability AI Community License | 允许商用（需注册 + NOTICE） |
| ControlNet | OpenRAIL-M | 允许商用 |
| IP-Adapter | Apache 2.0 | 允许商用 |
| FASHN VTON v1.5（模型权重 + fashn_vton 代码） | Apache 2.0 | 允许商用 |
| FASHN DWPose / YOLOX 权重 | Apache 2.0 | 允许商用 |
| fashn-human-parser（NVIDIA SegFormer 权重） | NVIDIA Source Code License（§3.3 含商用限制） | **禁止商用，商用路径注入零张量 stub 代替** |
| OpenPose | CC BY-NC-SA | 禁止商用 |
| DensePose | 内部使用 | 需确认 |

**推荐方案：**

1. **FASHN VTON v1.5（像素空间、无需 mask、TPS + 扩散）** 优先
   - Apache 2.0，允许商用；`segmentation_free=True` 强制关闭人体解析
   - `LICENSE_MODE=commercial` 下 `backend/engine_fashn.py` 向 `sys.modules` 注入
     零张量 `FashnHumanParser` stub，NVIDIA SegFormer 权重永不下载、永不加载
   - 消费级 GPU 可跑（约 972M 参数）；大字母/图案仍走确定性 `pattern_lock` 贴印

2. **SDXL + ControlNet Inpaint + IP-Adapter** 兜底
   - 全部允许商用
   - Inpaint 权重缺失时回退全图 ControlNet 并标记 degraded
   - 需要自行训练换衣适配器；训练数据必须来源合法

3. **联系 IDM-VTON 原作者**
   - 申请商业授权
   - 可能收费或拒绝
   - 需要书面协议

3. **使用商业 API**
   - 如 Replicate、Hugging Face Inference
   - 付费使用，合规风险由服务商承担
   - 成本高，但无需自行部署

印花/logo/文字保真的分层修复（分辨率门闩、两段 inpaint、OpenCV TPS 贴印、OCR 质检）只允许走路径 B 的组件。完整约束见当前工作区 `.monkeycode/specs/print-fidelity-commercial-roadmap/` 与 `docs/SDXL_PIPELINE.md`「印花保真商用技术路线」。IDM-VTON GarmentNet、DensePose 权重、以及 fashn-human-parser（NVIDIA SegFormer 权重）不进入商用加载集；FASHN VTON v1.5 主模型权重与 `fashn_vton` 代码为 Apache 2.0，允许商用。

### 路径 C：混合策略

- 开发/测试阶段：使用 IDM-VTON（research 模式）
- 生产阶段：切换到商用模型（commercial 模式）
- 通过 `LICENSE_MODE` 环境变量切换

## Stability AI Community License 合规（SDXL 基础模型）

商用 pipeline 的 SDXL base 模型遵循 **Stability AI Community License**（Last Updated 2026-07-05）。商用目的使用或分发前，必须同时满足以下义务：

| 义务 | 要求 | 本项目落地 |
|------|------|------------|
| 注册 | 商用目的使用或分发前，必须在 Stability AI 完成免费注册 | 上线前填注册表单 |
| 附协议副本 | 向第三方分发时附带本协议副本 | 随产品 / 服务分发协议副本 |
| Notice 文件 | 随分发附带 Notice 文件，文本固定：`This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved` | 仓库根目录 `NOTICE` |
| 显著署名 | 在相关网站、用户界面、about 页或产品文档显著展示 `Powered by Stability AI` | 前端页脚 + README |

**收入门槛**：你与关联公司合计年收入低于 $100 万时，商用免费但必须注册；超过门槛协议自动终止，需另行申请付费 Enterprise License。

> 注意：ControlNet / IP-Adapter 等其余组件各自遵循其许可证；IDM-VTON 的 CC BY-NC-SA 禁令与 Stability 协议无关，商用仍须替换或购买授权。

## 运行时门禁

`backend/engine.py` 根据 `LICENSE_MODE` 环境变量决定引擎可用性：

| 模式 | 环境变量值 | IDM-VTON 可用性 | 商用合规 |
|------|-----------|----------------|----------|
| 研究 | `research` | 可用（需 GPU） | 不合规 |
| 商业 | `commercial` | 禁用 | 合规（需商用模型） |
| 未设置 | 空 | 禁用 | 需确认 |

## 配置示例

### 研究模式（默认）

```bash
export LICENSE_MODE=research
# IDM-VTON 可用，但不得用于生产
```

### 商业模式

```bash
export LICENSE_MODE=commercial
# IDM-VTON 禁用，必须使用商用模型
# 需自行实现或接入商用 API
```

## 合规检查清单

- [ ] 确认使用场景（研究/商业）
- [ ] 设置 `LICENSE_MODE` 环境变量
- [ ] 检查 `/api/license` 返回的合规状态
- [ ] 研究模式：确保不面向真实用户
- [ ] 商业模式：替换为商用模型或 API
- [ ] 保留所有授权文件（LICENSE.txt）
- [ ] 署名（BY）：在产出物中标注来源
- [ ] Stability AI：完成商用注册、附带协议副本与 NOTICE 文件
- [ ] Stability AI：在界面 / 文档显著展示 `Powered by Stability AI`

## 责任声明

本项目提供引擎切换框架，但不保证任何特定模型的合规性。

使用前必须：
1. 阅读并理解所有模型许可证
2. 咨询法律专业人士
3. 获取必要的商业授权

**本项目作者不承担因违反模型许可证导致的法律责任。**
