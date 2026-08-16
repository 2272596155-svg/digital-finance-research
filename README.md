# AI 金融科技上市公司业务布局与人才需求研究

本仓库保存课程作业中招聘岗位 Prompt 工程实验的可复现代码、确定性实验输入与 440-v2 Benchmark 结果，同时提供金融岗位 AI 替代风险评分、SmallTransformer 产品线分类微调和 LinUCB 岗位推荐排序三项补充实验代码。

研究数据来自公开招聘页面及公司官网等公开来源。Prompt 实验用于比较三类 Prompt 在招聘信息结构化抽取任务中的准确性、稳定性、幻觉控制和运行效率；补充实验用于检验岗位风险评分的一致性、产品线分类的自动化能力以及岗位推荐排序方法。

## 1. 研究范围

* 原始岗位表：440 行。
* 清洗后有效岗位：439 条，覆盖 71 家公司。
* 平衡 Benchmark：88 条，其中 AI 相关岗位44条、非 AI 岗位44条。
* Prompt：P1、P2、P3。
* 重复次数：每个岗位与 Prompt 组合重复3次。
* 计划与实际调用：88 × 3 × 3 = 792次，全部成功。
* 模型：`deepseek-v4-flash`。
* 温度：0；最大输出：1200 tokens。
* 输入基础：岗位结构化字段与证据摘要，不包含完整职位描述。
* 参考答案：规则生成并待人工复核的银标准。

除 Prompt 实验外，仓库还提供以下三项补充实验代码：

1. 基于S/D/C/G/H五个维度的金融岗位AI替代风险评分及双评分者一致性验证；
2. 基于招聘岗位数据的SmallTransformer八类产品线分类微调；
3. 基于用户画像和模拟反馈的LinUCB岗位推荐排序及离线评价。

## 2. 仓库结构

```text
digital-finance-research/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ 440条岗位数据_整理版.xlsx
├─ prompt_experiment_runner_440_v2.py
├─ recruitment_prompt_input_440_v2.json
├─ verify_results_440_v2.py
├─ experiments_code.py
└─ runs_440_v2/
   ├─ preflight_report.json
   ├─ benchmark_metrics.json
   └─ benchmark_runs.jsonl
```

Prompt 核心实验从 `recruitment_prompt_input_440_v2.json` 开始复现。该文件包含确定性的439条实验输入、88条 Benchmark ID、P1/P2/P3原文、参考答案及随机种子。Excel岗位表用于核对数据来源和清洗结果。

`experiments_code.py` 汇总金融岗位AI替代风险评分、SmallTransformer微调和LinUCB推荐排序三项补充实验。

Prompt设计文档采用Markdown格式另行提交，不作为本代码仓库中的必需文件。

## 3. 运行环境

* Python 3.10或更高版本。
* Prompt runner仅使用Python标准库。
* 调用DeepSeek API时需要在本机配置环境变量 `DEEPSEEK_API_KEY`。
* 补充实验需要安装 `numpy`、`pandas`、`torch` 和 `scikit-learn`。

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

macOS/Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="你的密钥"
```

如果 `requirements.txt` 中没有包含补充实验依赖，可以直接安装：

```bash
pip install numpy pandas torch scikit-learn
```

不要把真实API Key写入任何仓库文件、运行结果或聊天记录。`.env.example` 只保留空变量名。

## 4. Prompt实验复现步骤

### 4.1 离线核验已有结果

该步骤不联网，也不会调用DeepSeek：

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

预检检查输入岗位数、Benchmark数量、AI/非AI平衡、Prompt ID、唯一岗位ID、参考答案结构、空文本和环境变量状态。预检不会发出网络请求。

### 4.3 重新运行Benchmark

警告：该命令会真实调用API并产生费用。为了避免重复调用，建议先把已有 `runs_440_v2` 目录备份，或者指定一个新的输出目录。

```bash
python prompt_experiment_runner_440_v2.py \
  --mode benchmark \
  --repeats 3 \
  --output-dir runs_440_v2_reproduction
```

Windows PowerShell可写成一行：

```powershell
python prompt_experiment_runner_440_v2.py --mode benchmark --repeats 3 --output-dir runs_440_v2_reproduction
```

Runner使用唯一 `run_id` 支持续跑；同一输出目录中已经完成的运行会自动跳过，从而降低重复调用和重复计费风险。

## 5. 补充实验代码

### 5.1 金融岗位AI替代风险评分

补充代码按照以下公式计算金融岗位AI替代风险：

```text
AI替代风险分 =
0.25 × S
+ 0.20 × D
+ 0.20 × C
+ 0.15 × G
+ 0.20 × (100 − H)
```

其中：

* `S`：任务标准化程度；
* `D`：数据数字化程度；
* `C`：判断规则可编码程度；
* `G`：生成式AI支持程度；
* `H`：对人类责任、信任、协商和复杂判断的依赖程度。

代码同时计算双评分者的Cohen’s Kappa、加权Kappa、ICC和平均绝对差等一致性指标，并输出岗位风险评分结果。

### 5.2 SmallTransformer产品线分类微调

SmallTransformer实验将招聘岗位文本映射至以下八类AI金融产品线：

1. AI平台与模型服务；
2. 数据平台与基础设施；
3. 智能投顾与财富管理；
4. 交易与清算系统；
5. 风控与合规系统；
6. 数字银行与渠道；
7. 智能客服与营销；
8. 金融信息安全。

实验将SmallTransformer与TF-IDF+逻辑回归、TF-IDF+MLP两类基线模型进行比较，并输出分类准确率、F1、分类报告和模型权重。

### 5.3 LinUCB岗位推荐排序

LinUCB实验基于岗位类别、AI属性、薪资、地点、经验和学历等特征构建岗位表示，并结合用户画像模拟岗位点击、收藏、申请和忽略等反馈。

实验使用多个随机种子重复运行，并通过以下指标评价推荐排序效果：

* NDCG@10；
* Recall@10；
* IPS估计；
* DR（Doubly Robust）估计；
* 与随机推荐策略的比较结果。

该实验属于课程研究中的离线模拟，用于展示强化学习与上下文老虎机在岗位推荐排序中的应用，不代表招聘平台真实线上推荐结果。

### 5.4 运行补充实验

运行前需要将代码使用的岗位JSON数据文件放在 `experiments_code.py` 同一目录，并确保文件名与脚本中的数据加载配置一致。

执行命令：

```bash
python experiments_code.py
```

程序将依次执行三项实验，并保存：

* 双评分者一致性检验结果；
* SmallTransformer模型权重与分类指标；
* LinUCB推荐排序指标与模拟交互结果。

## 6. Prompt评价指标

* **标量准确率**：岗位名称、薪资、地点、学历和经验五个标量字段标准化后的精确匹配率。
* **List F1**：硬技能、软技能和AI技术栈在参考列表非空时的集合级F1。
* **幻觉率**：模型输出中既不在参考答案、也无法由输入证据直接支持的非空原子值占比。
* **稳定性**：同一岗位、同一Prompt三次输出两两Jaccard相似度的均值。
* **Schema合规率**：输出可以解析为JSON，且必需字段和数据类型符合约定的比例。
* **平均耗时**：成功API请求的端到端平均秒数。
* **综合分**：按脚本中预设权重综合标量准确率、List F1、幻觉控制、稳定性、Schema合规率和速度。

List F1只在参考列表非空时计算。本轮可评价运行数为108，其中 `hard_skills` 覆盖60次、`ai_stack` 覆盖69次、`soft_skills` 覆盖0次。因此，List F1不能解释为全部技能字段的总体准确率。

## 7. Benchmark结果

| Prompt   |      标量准确率 | List F1 |       幻觉率 |        稳定性 | Schema合规率 |   平均耗时 |       综合分 |
| -------- | ---------: | ------: | --------: | ---------: | --------: | -----: | --------: |
| P1 直接抽取  |     89.39% |  28.30% |     2.04% |     98.65% |    99.62% | 1.112秒 | **81.75** |
| P2 证据约束  | **95.83%** |  14.47% | **0.20%** |     98.48% |  **100%** | 1.459秒 |     80.18 |
| P3 双阶段自检 |     93.86% |  27.15% |     0.81% | **98.84%** |  **100%** | 1.635秒 |     81.62 |

按预设综合权重，P1以81.75分排名第一；P3仅落后0.13分；P2在标量准确率和幻觉控制上最优，但列表召回较低。因此实际应用采用分层路由：

1. **P1主抽取**：用于批量基础抽取。
2. **P2证据闸门**：用于高风险、合规或要求“宁缺毋滥”的字段。
3. **P3异常复核**：仅在字段冲突、证据不足或Schema异常时触发。

## 8. 完整性与安全说明

* `benchmark_runs.jsonl` 共792行，对应792个唯一 `run_id`。
* 792次调用全部成功；791条通过Schema校验，唯一不合规记录来自P1。
* 结果文件不包含 `DEEPSEEK_API_KEY` 的值。
* API Key仅通过环境变量读取，脚本不会打印或写入密钥。
* 招聘页面具有时间快照属性；未检出岗位不能解释为企业没有相关能力。
* 招聘URL可能是公司招聘主页，一个URL对应多个岗位不等于岗位重复。
* 银标准仍需人工抽检，不能直接等同于完整JD的独立人工金标准。
* LinUCB实验属于离线模拟，不应解释为真实招聘平台的线上推荐效果。
* 模型输出和风险评分只用于课程研究，不应作为真实招聘、授信或职业决策的唯一依据。

## 9. 主要文件说明

* `prompt_experiment_runner_440_v2.py`：DeepSeek API调用、断点续跑、Schema校验和Prompt指标计算代码。
* `recruitment_prompt_input_440_v2.json`：确定性实验输入、Prompt原文、Benchmark ID和银标准。
* `verify_results_440_v2.py`：不联网核验输入数据及已有Prompt实验结果的完整性。
* `experiments_code.py`：金融岗位AI替代风险评分、SmallTransformer微调和LinUCB岗位推荐排序的补充实验代码。
* `440条岗位数据_整理版.xlsx`：招聘岗位原始整理表和汇总统计。
* `runs_440_v2/benchmark_metrics.json`：Prompt实验汇总指标和分组指标。
* `runs_440_v2/benchmark_runs.jsonl`：792条逐次运行记录。
* `runs_440_v2/preflight_report.json`：运行前的数据结构与安全预检报告。
* `Prompt设计文档.md`：采用Markdown格式另行提交，不属于本仓库文件。

## 10. 研究伦理与用途

本仓库仅用于课程研究、方法展示和结果复现。招聘信息来自公开页面，可能随时间变化。研究结论反映的是采集时点的公开信息和招聘信号，不应被解释为对企业、岗位或求职者的自动化最终判断。

年报、官网、招聘信息和模型输出分别反映正式披露、产品化情况、当期人才需求和自动化分析结果，不能相互简单替代。人工复核仍然是事实核验、异常处理和最终责任判断的重要环节。
