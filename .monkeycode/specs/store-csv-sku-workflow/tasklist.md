# 需求实施计划: store-csv-sku-workflow

- [x] 1. 数据模型变更：Store 增加 default_sku，Job 增加 bound_store_id
  - [x] 1.1 StoreBody / StorePatch 增加 `default_sku: str | None = ""` 字段（main.py:196）
  - [x] 1.2 store.py 的 `add_store` meta 字典补 `default_sku`（main.py:417）
  - [x] 1.3 batch 创建 job 时接受可选 `bound_store_id` 参数，写入 job（main.py batch 端点 + store.py）
  - [x] 1.4 推送端点 `push_job` 默认勾选 `bound_store_id`（main.py:750 store_ids 回退逻辑）
  - [x]* 1.5 单元测试：store 创建/读取含 default_sku；job 含 bound_store_id；推送默认勾选

- [x] 2. 后端：CSV + 图片 ZIP 批量导出 API
  - [x] 2.1 新建 `backend/export_csv.py`：`export_jobs_csv(job_ids) -> bytes`，生成 CSV + 复制成品图到 ZIP
    - CSV 列：`job_id,image_filename,source_name,store_name,store_id,product_id,sku,status`
    - 图片按 `{job_id}.jpg` 命名复制进 ZIP
    - job 不存在或未 done 时跳过
    - 引用需求 Req2
  - [x] 2.2 main.py 新增 `POST /api/jobs/export-csv`，入参 `job_ids: list[str]`，返回 `StreamingResponse`（media_type=application/zip）
  - [x]* 2.3 单元测试：3 个 done 任务导出 ZIP，验证 CSV 行数=图片数、image_filename 一致、未绑定店铺 store 列为空

- [x] 3. 后端：CSV + 图片批量推送 API（店铺详情页入口）
  - [x] 3.1 main.py 新增 `POST /api/stores/{store_id}/batch-push`，multipart 接收 CSV 文件 + 图片文件列表
  - [x] 3.2 解析 CSV 为行数组，校验列名（缺列返回 400 列出期望列名）
    - 引用需求 Req3
  - [x] 3.3 逐行处理：
    - store_id 缺失 → 填入当前店铺 ID
    - image_filename 在上传包中存在 → 用上传替换图；否则用原成品图；都不存在 → 跳过标注「图片缺失」
    - product_id/sku 缺失 → 回退店铺 default_product_id / default_sku
  - [x] 3.4 调 `push_store(job, store, {sku, product_id})` 逐行推送，返回 `[{job_id, ok, message, url}]`
  - [x]* 3.5 单元测试：上传 CSV + 2 替换图验证替换图优先；缺图行跳过标注；SKU 回退链（行>店铺>空）

- [x] 4. 后端：推送选择面板数据接口
  - [x] 4.1 `/api/stores` 列表响应已含 product_id，补 `default_sku` 字段返回（main.py _mask_store）
  - [x] 4.2 推送结果 `push_results` 增加 `product_id` / `sku` 记录（main.py:788 push_job 返回值）
  - [x]* 4.3 单元测试：store 列表含 default_sku；push_results 记录各自 product_id/sku

- [x] 5. 检查点 - 后端全量验证
  - 确保 `ruff check backend/` + `ruff format --check` 干净
  - 确保 `pytest tests/ -q` 全绿

- [ ] 6. 前端：店铺列表页改动
  - [ ] 6.1 店铺卡片增加「换衣」按钮，点击跳转 `/dashboard?store={store_id}`（src/main.js 店铺渲染）
  - [ ] 6.2 店铺卡片增加「批量推送」按钮，点击打开 CSV 上传弹窗
  - [ ] 6.3 dashboard 页读取 URL 参数 `store`，batch 创建时传 `bound_store_id`

- [ ] 7. 前端：任务列表导出 CSV
  - [ ] 7.1 任务列表批量勾选后显示「导出 CSV」按钮（src/main.js 队列面板）
  - [ ] 7.2 点击调 `POST /api/jobs/export-csv`（fetch + blob），触发浏览器下载 ZIP

- [ ] 8. 前端：推送选择面板
  - [ ] 8.1 单个任务「推送」改为弹出选择面板（替代原直接推送）
  - [ ] 8.2 面板列出所有店铺，每行：勾选框 + 店铺名 + product_id 输入框 + sku 输入框（预填 default 值）
  - [ ] 8.3 勾选多店铺各自修改 product_id/sku，提交调 `POST /api/jobs/{id}/push` 带 store_ids + 每店 sku/product_id
  - [ ] 8.4 push_job 端点支持每店独立 sku/product_id（main.py PushBody 改为 `store_overrides: dict[store_id, {sku, product_id}]`）

- [ ] 9. 前端：CSV 上传弹窗（店铺详情页）
  - [ ] 9.1 弹窗支持拖入 CSV 文件 + 多个图片文件
  - [ ] 9.2 解析 CSV 预览表格（job_id / image_filename / sku / 状态列）
  - [ ] 9.3 点击「批量推送」→ `POST /api/stores/{id}/batch-push`（FormData 带 CSV + 图片）
  - [ ] 9.4 返回结果表格显示每行 ok/message

- [ ] 10. 检查点 - 前端全量验证
  - 确保 `npm test` 全绿（14 个）
  - 确保 `npm run build` 通过
  - 确保 `ruff check` + `pytest` 仍全绿
  - 如有疑问询问用户
