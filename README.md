# AI 金融科技上市公司业务布局与人才需求研究

本仓库保存课程作业中招聘岗位 Prompt 工程实验的可复现代码、确定性实验输入与 440-v2 Benchmark 结果。研究数据来自公开招聘页面及公司官网等公开来源，实验用于比较三类 Prompt 在招聘信息结构化抽取任务中的准确性、稳定性、幻觉控制和运行效率。

## 1. 研究范围

- 原始岗位表：440 行。
- 清洗后有效岗位：439 条，覆盖 71 家公司。
- 平衡 Benchmark：88 条，其中 AI 相关岗位 44 条、非 AI 岗位 44 条。
- Prompt：P1、P2、P3。
- 重复次数：每个岗位与 Prompt 组合重复 3 次。
- 计划与实际调用：88 × 3 × 3 = 792 次，全部成功。
- 模型：`deepseek-v4-flash`。
- 温度：0；最大输出：1200 tokens。
- 输入基础：岗位结构化字段与证据摘要，不包含完整职位描述。
- 参考答案：规则生成并待人工复核的银标准。

本轮只完成 Benchmark，没有执行 439 条岗位的 `full` 模式。仓库不提供也不宣称存在 439 条全量抽取结果。

## 2. 仓库结构

```text
digital-finance-research/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ prompt_experiment_runner_440_v2.py
├─ recruitment_prompt_input_440_v2.json
├─ verify_results_440_v2.py
├─  440条岗位数据_整理版.xlsx
└─ runs_440_v2/
   ├─ preflight_report.json
   ├─ benchmark_metrics.json
   └─ benchmark_runs.jsonl
```

核心实验从 `recruitment_prompt_input_440_v2.json` 开始复现。该文件包含确定性的 439 条实验输入、88 条 Benchmark ID、P1/P2/P3 原文、参考答案及随机种子。Excel 岗位表用于核对数据来源和清洗结果。

## 3. 运行环境

- Python 3.10 或更高版本。
- Prompt runner 仅使用 Python 标准库，无需安装额外第三方包。
- 调用 DeepSeek API 时需要在本机配置环境变量 `DEEPSEEK_API_KEY`。

可选的虚拟环境配置：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY="你的密钥"
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="你的密钥"
```

不要把真实 API Key 写入任何仓库文件、运行结果或聊天记录。`.env.example` 只保留空变量名。

## 4. 复现步骤

### 4.1 离线核验已有结果

该步骤不联网，也不会调用 DeepSeek：

```bash
python verify_results_440_v2.py
```

预期输出应包含：

```text
PASS: 440-v2 reproducibility checks completed
```

### 4.2 运行预检

```bash
python prompt_experiment_runner_440_v2.py --mode preflight
```

预检检查输入岗位数、Benchmark 数量、AI/非 AI 平衡、Prompt ID、唯一岗位 ID、参考答案结构、空文本和环境变量状态。预检不会发出网络请求。

### 4.3 重新运行 Benchmark

警告：该命令会真实调用 API 并产生费用。为了避免重复调用，建议先把已有 `runs_440_v2` 目录备份或指定一个新的输出目录。

```bash
python prompt_experiment_runner_440_v2.py \
  --mode benchmark \
  --repeats 3 \
  --output-dir runs_440_v2_reproduction
```

Windows PowerShell 可写成一行：

```powershell
python prompt_experiment_runner_440_v2.py --mode benchmark --repeats 3 --output-dir runs_440_v2_reproduction
```

Runner 使用唯一 `run_id` 支持续跑；同一输出目录中已经完成的运行会自动跳过，降低重复计费风险。

## 5. 评价指标

- **标量准确率**：岗位名称、薪资、地点、学历和经验五个标量字段标准化后的精确匹配率。
- **List F1**：硬技能、软技能和 AI 技术栈在参考列表非空时的集合级 F1。
- **幻觉率**：模型输出中既不在参考答案、也无法由输入证据直接支持的非空原子值占比。
- **稳定性**：同一岗位、同一 Prompt 三次输出两两 Jaccard 相似度的均值。
- **Schema 合规率**：输出可以解析为 JSON，且必需字段和数据类型符合约定的比例。
- **平均耗时**：成功 API 请求的端到端平均秒数。
- **综合分**：按脚本中预设权重综合标量准确率、List F1、幻觉控制、稳定性、Schema 合规率和速度。

List F1 只在参考列表非空时计算。本轮可评价运行数为 108，其中 `hard_skills` 覆盖 60 次、`ai_stack` 覆盖 69 次、`soft_skills` 覆盖 0 次。因此 List F1 不能解释为全部技能字段的总体准确率。

## 6. Benchmark 结果

| Prompt | 标量准确率 | List F1 | 幻觉率 | 稳定性 | Schema 合规率 | 平均耗时 | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 直接抽取 | 89.39% | 28.30% | 2.04% | 98.65% | 99.62% | 1.112 秒 | **81.75** |
| P2 证据约束 | **95.83%** | 14.47% | **0.20%** | 98.48% | **100%** | 1.459 秒 | 80.18 |
| P3 双阶段自检 | 93.86% | 27.15% | 0.81% | **98.84%** | **100%** | 1.635 秒 | 81.62 |

按预设综合权重，P1 以 81.75 分排名第一；P3 仅落后 0.13 分；P2 在标量准确率和幻觉控制上最优，但列表召回较低。因此实际应用采用分层路由：

1. **P1 主抽取**：用于批量基础抽取。
2. **P2 证据闸门**：用于高风险、合规或要求“宁缺毋滥”的字段。
3. **P3 异常复核**：仅在字段冲突、证据不足或 Schema 异常时触发。

## 7. 完整性与安全说明

- `benchmark_runs.jsonl` 共 792 行，对应 792 个唯一 `run_id`。
- 792 次调用全部成功；791 条通过 Schema 校验，唯一不合规记录来自 P1。
- 结果文件不包含 `DEEPSEEK_API_KEY` 的值。
- API Key 仅通过环境变量读取，脚本不会打印或写入密钥。
- 招聘页面具有时间快照属性；未检出岗位不能解释为企业没有相关能力。
- 招聘 URL 可能是公司招聘主页，一个 URL 对应多个岗位不等于岗位重复。
- 银标准仍需人工抽检，不能直接等同于完整 JD 的独立人工金标准。

## 8. 主要文件说明

- `Prompt设计与运行说明.md`：三套 Prompt 原文、设计差异、实验控制、指标和结果解释。
- `prompt_experiment_runner_440_v2.py`：API 调用、断点续跑、Schema 校验和指标计算代码。
- `recruitment_prompt_input_440_v2.json`：确定性实验输入、Prompt、Benchmark ID 和银标准。
- `verify_results_440_v2.py`：不联网核验输入及已有结果完整性。
- `runs_440_v2/benchmark_metrics.json`：汇总指标和分组指标。
- `runs_440_v2/benchmark_runs.jsonl`：792 条逐次运行记录。
- `runs_440_v2/preflight_report.json`：运行前数据结构与安全预检报告。

## 9. 研究伦理与用途

本仓库仅用于课程研究和方法复现。招聘信息来自公开页面，可能随时间变化。研究结论是对采集时点公开信号的分析，不应被解释为对企业、岗位或求职者的自动化最终判断。

