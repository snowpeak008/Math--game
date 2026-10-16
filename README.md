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
  - **已知限制**：beam / MCTS / 分支定界（BnB）优化器尚不支持多货币；MCTS 运行较慢。
- **0.1.2** — 贪心机会成本实验选项（默认关闭，实测无一致收益，未采纳）。
