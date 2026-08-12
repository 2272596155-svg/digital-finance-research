# 招聘岗位 Prompt 工程实验

本目录用于复现“AI 金融科技上市公司的业务布局与人才需求分析”中的招聘信息结构化实验。

## 实验范围

- 公司宇宙：72 家，全部保留；没有岗位样本的公司继续标记为无公开招聘信号。
- 全量代表岗位：128 条。
- 分层评测集：24 条，其中 AI 相关与非 AI 对照各 12 条，保留 4 条低质量证据和 6 条中质量证据。
- Prompt：P1 直接零样本 JSON 抽取、P2 Schema 与证据约束、P3 抽取—核验双阶段自检。
- 重复运行：每条岗位、每套 Prompt 运行 3 次，共计划 216 次 API 调用。

## 安全配置

API 密钥只从环境变量 `DEEPSEEK_API_KEY` 读取。不要把密钥写入脚本、工作簿或聊天。

默认模型为 `deepseek-v4-flash`，接口地址为 `https://api.deepseek.com`。模型和地址以 2026-08-12 的 DeepSeek 官方 API 文档为准。

## 命令

离线预检（不联网、不消耗额度）：

```bash
python prompt_experiment_runner.py --mode preflight
```

配置环境变量后运行 24 条分层评测集：

```bash
python prompt_experiment_runner.py --mode benchmark --repeats 3
```

选出最佳 Prompt 后全量处理 128 条岗位，例如使用 P3：

```bash
python prompt_experiment_runner.py --mode full --prompt-id P3 --repeats 1
```

## 评价指标

- 标量字段准确率：职位、薪资、地点、学历、经验的标准化精确匹配率。
- 列表字段 F1：硬技能、软技能、AI 技术栈的集合级 Macro-F1。
- 幻觉率：无法由输入文本或银标准支持的非空原子值占比。
- 稳定性：相同岗位、相同 Prompt 三次输出的两两集合 Jaccard 均值。
- Schema 合规率：JSON 可解析且公共字段名、数据类型完整的运行占比。
- 平均耗时：成功 API 请求的端到端秒数。

当前标签属于“人工复核的银标准”，正式报告使用前仍应对高影响误差做人工抽检。
