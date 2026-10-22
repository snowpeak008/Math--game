# Math--game

放置 / 增量（idle / incremental）游戏**数值平衡工具**。用于建模资源产出、升级价格曲线、多阶段进度，并模拟、分析游戏节奏。

本项目基于 [ac2522/IdleFramework](https://github.com/ac2522/IdleFramework)（作者 ac2522）。0.1.0 版本即上游原始代码（提交 `a0447d5`），在此致谢原作者。

## 许可证说明

上游仓库在该提交中**未包含 LICENSE 文件**，许可状态不明确。在获得原作者明确授权前，请勿将本代码视为开源许可代码进行再分发或商用。

## CI 说明

上游的 GitHub Actions CI 配置（`.github/workflows/ci.yml`）未包含在本仓库中，后续可按需重新添加。

## 快速开始

需要 Python 3.11+ 与 Node.js 18+。

```bash
# 安装依赖
pip install -e ".[dev,server]"
cd frontend && npm install && cd ..

# 启动后端（FastAPI，端口 8000）
uvicorn server.app:app --reload --port 8000
# 或：make server

# 另开终端，启动前端（Vite 开发服务器）
cd frontend && npm run dev
# 或：make frontend
```

运行测试：`pytest tests/ -v`（前端：`cd frontend && npm test`）。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `src/` | 核心引擎：游戏模型、模拟与分析 |
| `server/` | FastAPI 后端（`server.app:app`） |
| `frontend/` | Vite + React 前端界面 |
| `tests/` | 单元 / 属性 / E2E 测试 |
| `benchmarks/` | 性能基准 |
| `docs/` | 文档 |
| `samples/` | 示例游戏配置与分析结果（米粒测试版、多货币商店示例） |
| `screenshots/` | 界面截图 |
| `IDLE_GAME_MECHANICS_RESEARCH.md` | 放置游戏机制调研 |

## 版本历史

- **0.1.0** — 上游基线：IdleFramework `a0447d5` 原始代码，新增中文 README 与 `.gitignore`；移除测试产物与 CI 配置（`.github/workflows/`）。
- **0.1.1** — 在上游基础上的首个功能版本：
  - **中文输入界面**：`/input` 页面、表格粘贴导入、校验 API（validate）。
  - **引擎修复**：`advance_to` 结果与步长粒度无关；支持零产出起步；容量与静态检查（static checks）；贪心策略按回本时间（payback）决策。
  - **存量联动闭式解**：与逐步模拟相比误差由 23%–212% 降至 <0.01%。
  - **多货币**：成本列表支持多种货币、`requires` 解锁阈值、按货币分账（ledger）。
  - **示例**：`samples/米粒测试版`、`samples/多货币商店示例`。
  - **中文文档**。
  - **已知限制**：beam / MCTS / 分支定界（BnB）优化器尚不支持多货币（已于 0.1.4 解除）；MCTS 运行较慢。
- **0.1.2** — 贪心机会成本实验选项（默认关闭，实测无一致收益，未采纳）。
- **0.1.3** — 引擎热点缓存，MCTS 测试 475s→185s，结果不变。
- **0.1.4** — beam/MCTS/BnB 支持多货币价格与购买门槛（解除 0.1.1 中的对应已知限制）。
- **0.1.5** — 优化器与引擎自动购买隔离 + 预留账本；beam 时间同步排序；BnB 上界修正并与穷举对齐；MCTS 实时等待；机会成本自动推导实验（不采纳）；新增 `docs-zh/优化器说明.md`。
- **0.1.6** — beam 改为 greedy 补全打分（结果 ≥ greedy，默认宽度 5）；MCTS 冻结为实验性；存量溢出改为 log 空间计算并抛出 `StockOverflowError`，新增无界增长静态警告；串行关卡掉落节点；死锁规则（无可达来源才报）；约束货币包含 `requires`；比较规则 Δ收入/c > k。
- **0.1.7** — 影子价格估值（cost/requires/战力分开定义，默认开启，可用 valuation=legacy）；溢出后以 ln 状态继续模拟；刷关消融实验；beam 宽度对照表；比较规则数据。
- **0.1.8** — 新增 `docs-zh/lessons/铸剑大师_经验包.md`：《铸剑大师》数值平衡经验包（参考经验，非唯一标准，供后续验收、开发测试对照）；纯文档，引擎代码不变。
- **0.1.9** — 同步 IdleFramework 引擎与分析改进（上游本地提交至 `8a96bda`）：新增 CMA-ES 逆向参数拟合（`inverse/`）；敏感度 / 临界点报告与瓶颈提示（默认按节点到达时间）；经济-战力耦合、视野内可达性静态检查；升级前置 `prereq_all` / `prereq_any`（未满足时不可购买）及贪心开路估值；一次升级可加成多个目标（`effects`）、乘数可限定作用范围（`targets`）、多货币线性估值权重（`valuation_weights`）；greedy / beam「为 X 攒钱」、空档补买、乘数期望前瞻，beam 根取两种估值 greedy 的较优者，默认宽度交互 3 / 批量 5；目标函数字段（节点时间 / 产出率）；分析支持 `--max-steps` 和 JSON 钱包初始余额；移除机会成本实验选项及其测试；同步相关测试、实验脚本与 `docs-zh/优化器说明.md`。
- **0.1.10** — 新增 `docs-zh/lessons/挖到地心_经验包.md`：《挖到地心》（挖矿放置）数值平衡经验包（参考经验，非唯一标准，供后续验收、开发测试对照）；挖矿专属规则只放在项目附录，属游戏规则而非引擎标准；纯文档，引擎代码不变。
- **0.1.11** — 修订 `docs-zh/lessons/挖到地心_经验包.md`：定版改为 v5（候选 A7，4 案 2.89–3.20 小时），参数、门梯、时间线和验收清单按 v5 重写；v4 标为作废（2026-10-20 按真实存档校准单趟时长 / 产出后实际需 6.7–8.3 小时），并新增经验「锁定数值前先用真实存档校准单趟时长和产出」；纯文档，引擎代码不变。
