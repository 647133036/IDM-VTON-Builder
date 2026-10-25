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
| Stable Diffusion XL | OpenRAIL-M | 允许商用 |
| ControlNet | OpenRAIL-M | 允许商用 |
| IP-Adapter | Apache 2.0 | 允许商用 |
| OpenPose | CC BY-NC-SA | 禁止商用 |
| DensePose | 内部使用 | 需确认 |

**推荐方案：**

1. **SDXL + ControlNet + IP-Adapter** 组合
   - 全部允许商用
   - 需要自行训练换衣适配器
   - 训练数据必须来源合法

2. **联系 IDM-VTON 原作者**
   - 申请商业授权
   - 可能收费或拒绝
   - 需要书面协议

3. **使用商业 API**
   - 如 Replicate、Hugging Face Inference
   - 付费使用，合规风险由服务商承担
   - 成本高，但无需自行部署

### 路径 C：混合策略

- 开发/测试阶段：使用 IDM-VTON（research 模式）
- 生产阶段：切换到商用模型（commercial 模式）
- 通过 `LICENSE_MODE` 环境变量切换

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

## 责任声明

本项目提供引擎切换框架，但不保证任何特定模型的合规性。

使用前必须：
1. 阅读并理解所有模型许可证
2. 咨询法律专业人士
3. 获取必要的商业授权

**本项目作者不承担因违反模型许可证导致的法律责任。**
