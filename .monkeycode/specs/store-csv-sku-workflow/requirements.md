# Requirements Document: Store CSV/SKU Workflow

## Introduction

在现有换衣工作台基础上，增加「店铺维度换衣入口 + CSV 批量导出/导入 + SKU 级推送指定」三大功能，形成「换衣 → 导出 CSV → 人工修图 → CSV 批量推送」的完整工作流。解决换衣效果不理想时需要离线修图、再批量回传店铺的场景，以及推送时无法精确指定到具体商品/SKU 的问题。

## Glossary

- **Store**: 店铺，有独立 embed 链接与推送目标（Shopify/WordPress/其他）
- **SKU**: 商品库存单位，标识一个具体商品变体（如颜色/尺码组合）
- **Product ID**: 商品在店铺平台的唯一标识
- **CSV Manifest**: 包含成品图路径 + 店铺 + 商品/SKU 映射的批量推送清单
- **Push**: 将成品图上传到店铺平台的商品图位置

## Requirements

### Requirement 1: 店铺维度换衣入口

**User Story:** AS 店铺运营人员, I want 在店铺列表直接发起换衣任务, so that 换衣结果自动绑定到该店铺，省去后续推送时再选店铺的步骤

#### Acceptance Criteria

1. WHEN 用户在店铺列表点击「换衣」按钮, the system SHALL 跳转到换衣工作台并预选该店铺作为推送目标
2. WHEN 换衣任务在店铺维度发起时, the system SHALL 在 job 上记录 `bound_store_id`，推送时默认勾选该店铺
3. IF 用户在店铺维度发起换衣后想推送到其他店铺, the system SHALL 允许在推送时修改目标店铺列表

### Requirement 2: CSV + 图片批量导出

**User Story:** AS 店铺运营人员, I want 将一批换衣成品连同 CSV 清单一起导出, so that 换衣效果不理想时可以离线修图，修完后连图带 CSV 一起回传批量推送

#### Acceptance Criteria

1. WHEN 用户在任务列表勾选多个已完成任务并点击「导出 CSV」, the system SHALL 生成一个 ZIP 包，内含 CSV 清单 + 所有成品图文件
2. WHILE CSV 导出时, the system SHALL 在 CSV 中包含以下列：`job_id, image_filename, source_name, store_name, store_id, product_id, sku, status`
3. WHEN 用户下载 ZIP 后, the system SHALL 保证 CSV 中的 `image_filename` 列与 ZIP 内的实际图片文件名一一对应
4. IF 任务未绑定店铺, the system SHALL 在 CSV 的 store 列留空，允许用户离线填写
5. IF 任务未配置 SKU, the system SHALL 在 CSV 的 sku 列留空，允许用户离线填写

### Requirement 3: CSV + 图片批量导入推送（店铺维度入口）

**User Story:** AS 店铺运营人员, I want 在店铺详情页上传修改后的图片和 CSV, so that 一次性把几十张修好的图推送到该店铺的对应商品

#### Acceptance Criteria

1. WHEN 用户在店铺详情页点击「批量推送」并上传 CSV + 图片包, the system SHALL 解析 CSV 并为每一行创建一个推送任务
2. WHEN CSV 某行包含 `product_id` 或 `sku`, the system SHALL 用该值覆盖店铺默认值作为推送目标商品
3. WHEN CSV 某行的 `image_filename` 对应的图片在上传包中存在, the system SHALL 优先使用上传的替换图而非原成品图
4. IF CSV 某行缺少 `store_id`, the system SHALL 自动填入当前店铺的 ID（入口在店铺详情页）
5. IF CSV 某行的 `image_filename` 在上传包和原成品中都不存在, the system SHALL 跳过该行并在结果中标注「图片缺失」
6. WHEN CSV 批量推送完成, the system SHALL 返回每行的推送结果（ok/message/url）

### Requirement 4: 店铺 SKU 配置（默认 + 可覆盖）

**User Story:** AS 店铺管理员, I want 在店铺配置中增加默认 SKU, 且推送时可以临时覆盖, so that 既有默认值兜底又能按需指定不同 SKU

#### Acceptance Criteria

1. WHEN 创建或编辑店铺时, the system SHALL 提供一个 `default_sku` 字段输入框
2. WHILE 推送时未显式指定 SKU, the system SHALL 使用店铺的 `default_sku` 作为回退值
3. WHEN 推送时或 CSV 某行显式指定了 SKU, the system SHALL 用指定值覆盖店铺的 `default_sku`
4. WHEN 店铺配置了 `default_sku`, the system SHALL 在推送时将其传递给平台 API（Shopify variant / WordPress product）

### Requirement 5: 推送时选择店铺 + 商品/SKU

**User Story:** AS 店铺运营人员, I want 推送时能选择目标店铺和具体商品/SKU, so that 精确控制每张图推到哪个店铺的哪个商品

#### Acceptance Criteria

1. WHEN 用户对单个任务点击「推送」, the system SHALL 弹出一个选择面板，列出所有店铺及其 `default_product_id` / `default_sku`
2. WHEN 用户在选择面板中修改了某店铺的 product_id 或 sku, the system SHALL 用修改后的值覆盖该次推送
3. IF 用户在选择面板中勾选了多个店铺, the system SHALL 对每个店铺分别推送，各自使用其 product_id/sku
4. WHEN 推送完成, the system SHALL 在任务的 `push_results` 中记录每个店铺使用的 product_id/sku
