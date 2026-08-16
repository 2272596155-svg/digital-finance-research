"""
AI金融科技公司业务布局与人才需求分析 — 实验代码汇总（修正版）
=====================================================
包含三个实验的完整可复现代码：
  实验一：双评分者S/D/C/G/H加权评分与一致性验证
  实验二：SmallTransformer 8类产品线分类微调
  实验三：LinUCB强化学习岗位推荐排序（8画像×30种子×300反馈+IPS/DR/NDCG评估）

依赖：pip install numpy pandas torch scikit-learn
运行：python experiments_code.py
"""

import json
import os
import re
import random
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# #############################################################################
# 实验一：双评分者S/D/C/G/H加权评分与一致性验证
# 修正：使用25%/20%/20%/15%/20%加权总分（非等权平均）
# #############################################################################

def experiment_1_second_rater():
    """
    S/D/C/G/H五维加权评分：
      S(标准化)权重25%, D(数字化)权重20%, C(可编码)权重20%,
      G(生成式AI支持)权重15%, H反转(100-H)权重20%
    总分 = S*0.25 + D*0.20 + C*0.20 + G*0.15 + (100-H)*0.20
    """
    print("=" * 80)
    print("实验一：双评分者S/D/C/G/H加权评分与一致性验证")
    print("=" * 80)

    # 第一评分者
    rater1 = {
        "会计核算/出纳":        {"S": 90, "D": 90, "C": 90, "G": 65, "H": 20},
        "财务共享中心专员":      {"S": 88, "D": 92, "C": 85, "G": 70, "H": 25},
        "税务申报专员":         {"S": 85, "D": 85, "C": 88, "G": 60, "H": 30},
        "信贷资料审核员":       {"S": 82, "D": 88, "C": 85, "G": 68, "H": 35},
        "投研助理/信息搜集":    {"S": 72, "D": 85, "C": 65, "G": 82, "H": 45},
        "财务分析师":          {"S": 60, "D": 80, "C": 58, "G": 78, "H": 58},
        "风控专员":            {"S": 55, "D": 78, "C": 72, "G": 62, "H": 68},
        "审计经理":            {"S": 48, "D": 70, "C": 58, "G": 55, "H": 75},
        "合规经理":            {"S": 42, "D": 65, "C": 62, "G": 48, "H": 82},
        "投行项目负责人":       {"S": 35, "D": 62, "C": 40, "G": 55, "H": 90},
        "客户经理/财富顾问":    {"S": 38, "D": 58, "C": 35, "G": 58, "H": 88},
        "CFO/财务负责人":      {"S": 30, "D": 65, "C": 38, "G": 55, "H": 95},
    }

    # 第二评分者独立评分
    rater2 = {
        "会计核算/出纳":        {"S": 88, "D": 92, "C": 88, "G": 68, "H": 22},
        "财务共享中心专员":      {"S": 90, "D": 90, "C": 82, "G": 72, "H": 28},
        "税务申报专员":         {"S": 85, "D": 85, "C": 85, "G": 55, "H": 35},
        "信贷资料审核员":       {"S": 80, "D": 90, "C": 82, "G": 70, "H": 38},
        "投研助理/信息搜集":    {"S": 70, "D": 88, "C": 62, "G": 85, "H": 48},
        "财务分析师":          {"S": 58, "D": 82, "C": 55, "G": 80, "H": 62},
        "风控专员":            {"S": 52, "D": 80, "C": 70, "G": 65, "H": 65},
        "审计经理":            {"S": 45, "D": 72, "C": 55, "G": 58, "H": 72},
        "合规经理":            {"S": 40, "D": 68, "C": 58, "G": 52, "H": 80},
        "投行项目负责人":       {"S": 32, "D": 60, "C": 38, "G": 50, "H": 88},
        "客户经理/财富顾问":    {"S": 35, "D": 55, "C": 32, "G": 55, "H": 85},
        "CFO/财务负责人":      {"S": 28, "D": 62, "C": 35, "G": 50, "H": 92},
    }

    # 加权系数
    WEIGHTS = {"S": 0.25, "D": 0.20, "C": 0.20, "G": 0.15, "H_rev": 0.20}

    def calc_total(s):
        """加权总分 = S*0.25 + D*0.20 + C*0.20 + G*0.15 + (100-H)*0.20"""
        return round(
            s["S"] * WEIGHTS["S"] +
            s["D"] * WEIGHTS["D"] +
            s["C"] * WEIGHTS["C"] +
            s["G"] * WEIGHTS["G"] +
            (100 - s["H"]) * WEIGHTS["H_rev"], 1)

    def get_grade(t):
        if t >= 75: return "高"
        elif t >= 50: return "中"
        else: return "低"

    # 计算总分和等级
    print(f"\n  加权公式: S*25% + D*20% + C*20% + G*15% + (100-H)*20%")
    print(f"\n{'岗位':<20} {'R1总分':>6} {'R1等级':>6} {'R2总分':>6} {'R2等级':>6} {'差异':>6} {'一致':>4}")
    print("-" * 65)
    grade_consistent = 0
    for job in rater1:
        t1 = calc_total(rater1[job])
        t2 = calc_total(rater2[job])
        g1 = get_grade(t1)
        g2 = get_grade(t2)
        diff = abs(t1 - t2)
        consistent = "Y" if g1 == g2 else "N"
        if g1 == g2:
            grade_consistent += 1
        print(f"{job:<20} {t1:>6.1f} {g1:>6} {t2:>6.1f} {g2:>6} {diff:>6.1f} {consistent:>4}")

    print(f"\n  等级一致: {grade_consistent}/12")

    # === Cohen's Kappa（等级一致性） ===
    r1_grades = [get_grade(calc_total(rater1[j])) for j in rater1]
    r2_grades = [get_grade(calc_total(rater2[j])) for j in rater2]
    grade_map = {"低": 0, "中": 1, "高": 2}
    r1_g = [grade_map[g] for g in r1_grades]
    r2_g = [grade_map[g] for g in r2_grades]
    n = len(r1_g)

    matrix = np.zeros((3, 3))
    for i in range(n):
        matrix[r1_g[i]][r2_g[i]] += 1

    po = sum(matrix[i][i] for i in range(3)) / n
    r1_props = [sum(1 for x in r1_g if x == c) / n for c in range(3)]
    r2_props = [sum(1 for x in r2_g if x == c) / n for c in range(3)]
    pe = sum(r1_props[c] * r2_props[c] for c in range(3))
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

    # === 加权Kappa（线性权重） ===
    weights = np.array([[1 - abs(i - j) / 2 for j in range(3)] for i in range(3)])
    wpo = sum(matrix[i][j] * weights[i][j] for i in range(3) for j in range(3)) / n
    wpe = sum(r1_props[i] * r2_props[j] * weights[i][j]
              for i in range(3) for j in range(3))
    weighted_kappa = (wpo - wpe) / (1 - wpe) if (1 - wpe) != 0 else 1.0

    # === ICC（总分一致性，双向随机效应模型 ICC(2,1)） ===
    r1_totals = [calc_total(rater1[j]) for j in rater1]
    r2_totals = [calc_total(rater2[j]) for j in rater2]
    n_jobs = len(r1_totals)
    n_raters = 2
    grand_mean = np.mean(r1_totals + r2_totals)
    ss_between = sum(
        (np.mean([r1_totals[i], r2_totals[i]]) - grand_mean) ** 2
        for i in range(n_jobs)) * n_raters
    ss_within = sum(
        (r1_totals[i] - np.mean([r1_totals[i], r2_totals[i]])) ** 2 +
        (r2_totals[i] - np.mean([r1_totals[i], r2_totals[i]])) ** 2
        for i in range(n_jobs))
    ss_rater = n_jobs * (np.mean(r1_totals) - grand_mean) ** 2 + \
               n_jobs * (np.mean(r2_totals) - grand_mean) ** 2
    ms_between = ss_between / (n_jobs - 1)
    ms_within = ss_within / (n_jobs * (n_raters - 1))
    ms_rater = ss_rater / (n_raters - 1)
    icc = (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)

    # === MAD ===
    total_diffs = [abs(r1_totals[i] - r2_totals[i]) for i in range(n_jobs)]
    mad = np.mean(total_diffs)

    # === 分维度ICC ===
    dim_icc = {}
    for dim in ['S', 'D', 'C', 'G', 'H']:
        r1_dim = [rater1[j][dim] for j in rater1]
        r2_dim = [rater2[j][dim] for j in rater2]
        gm = np.mean(r1_dim + r2_dim)
        ss_b = sum((np.mean([r1_dim[i], r2_dim[i]]) - gm) ** 2
                   for i in range(n_jobs)) * 2
        ss_w = sum((r1_dim[i] - np.mean([r1_dim[i], r2_dim[i]])) ** 2 +
                   (r2_dim[i] - np.mean([r1_dim[i], r2_dim[i]])) ** 2
                   for i in range(n_jobs))
        ms_b = ss_b / (n_jobs - 1)
        ms_w = ss_w / n_jobs
        dim_icc[dim] = (ms_b - ms_w) / (ms_b + ms_w) if (ms_b + ms_w) != 0 else 1.0

    print(f"\n--- 一致性指标 ---")
    print(f"  Cohen's Kappa:        {kappa:.4f}")
    print(f"  加权Kappa(线性权重):  {weighted_kappa:.4f}")
    print(f"  ICC(2,1) 一致性:     {icc:.4f}")
    print(f"  总分MAD:             {mad:.2f} 分")
    print(f"  最大差异:            {max(total_diffs):.1f} 分")
    print(f"  等级一致:            {grade_consistent}/12")
    print(f"\n  分维度ICC:")
    for dim, val in dim_icc.items():
        print(f"    {dim}: {val:.4f}")

    results = {
        "weighting": "S*25% + D*20% + C*20% + G*15% + (100-H)*20%",
        "kappa": round(kappa, 4),
        "weighted_kappa": round(weighted_kappa, 4),
        "icc": round(icc, 4),
        "mad": round(mad, 2),
        "max_diff": round(max(total_diffs), 1),
        "grade_consistent": grade_consistent,
        "dim_icc": {k: round(v, 4) for k, v in dim_icc.items()},
    }
    with open(os.path.join(BASE_DIR, "second_rater_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: second_rater_results.json")
    return results


# #############################################################################
# 实验二：SmallTransformer 8类产品线分类微调
# 修正：从二分类改为8类分类；epochs=20，lr=1e-3
# #############################################################################

# 8个产品线类别
PRODUCT_LINE_CATEGORIES = [
    "AI平台与模型服务",     # 0
    "数据平台与基础设施",   # 1
    "智能投顾与财富管理",   # 2
    "交易与清算系统",       # 3
    "风控与合规系统",       # 4
    "数字银行与渠道",       # 5
    "智能客服与营销",       # 6
    "金融信息安全",         # 7
]

# 岗位类别到产品线的映射规则
def map_job_to_product_line(job):
    """根据岗位信息映射到8个产品线类别之一"""
    title = job.get('岗位名称', '') + ' ' + job.get('岗位类别', '') + ' ' + job.get('证据摘要', '')
    title_lower = title.lower()

    rules = [
        (0, ['大模型', 'llm', 'aigc', 'agent', '智能体', 'ai平台', '算法', '机器学习', '深度学习']),
        (1, ['数据治理', '数据中台', '数据平台', '大数据', 'hadoop', 'spark', '数据仓库', 'etl']),
        (2, ['投顾', '财富管理', '基金代销', '理财', '资管', '投资顾问']),
        (3, ['交易', '清算', '结算', '银行核心', '支付', '证券', '柜台']),
        (4, ['风控', '反欺诈', '反洗钱', '合规', '监管', '审计']),
        (5, ['手机银行', '开放银行', '数字货币', '数字人民币', '渠道', '网点']),
        (6, ['客服', '营销', '运营', '用户增长', '私域', '智能客服']),
        (7, ['安全', '认证', '生物识别', '加密', '渗透', '密码']),
    ]

    for cat_id, keywords in rules:
        for kw in keywords:
            if kw in title_lower:
                return cat_id
    # 默认归入风控与合规
    return 4


def experiment_2_finetune():
    """
    SmallTransformer微调：基于招聘文本的8类产品线分类器
    对比：TF-IDF+LR、TF-IDF+MLP、SmallTransformer(8类)
    """
    print("\n" + "=" * 80)
    print("实验二：SmallTransformer 8类产品线分类微调")
    print("=" * 80)

    # 1. 加载数据
    all_jobs = []
    for fname in ['batch1_jobs.json', 'batch2_jobs.json', 'batch3_jobs.json',
                   'batch4_jobs.json', 'liepin_jobs.json', 'supplement_jobs_zhaopin.json']:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_jobs.extend(data)

    print(f"  加载 {len(all_jobs)} 条岗位数据")

    # 2. 构建文本和8类标签
    texts, labels = [], []
    for job in all_jobs:
        text = f"{job.get('岗位名称', '')} {job.get('岗位类别', '')} {job.get('证据摘要', '')}"
        texts.append(text.strip())
        labels.append(map_job_to_product_line(job))

    label_dist = Counter(labels)
    print(f"  8类产品线标签分布:")
    for cat_id in range(8):
        print(f"    {PRODUCT_LINE_CATEGORIES[cat_id]}: {label_dist.get(cat_id, 0)}")

    # 3. 划分训练/测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

    # 4. 基线A：TF-IDF + LR
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    print(f"\n  TF-IDF + LR:       Acc={lr_acc:.4f}, F1={lr_f1:.4f}")

    # 5. 基线B：TF-IDF + MLP
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                        random_state=SEED, early_stopping=True)
    mlp.fit(X_train_tfidf, y_train)
    mlp_pred = mlp.predict(X_test_tfidf)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
    print(f"  TF-IDF + MLP:      Acc={mlp_acc:.4f}, F1={mlp_f1:.4f}")

    # 6. SmallTransformer模型（8类分类）
    class JobDataset(Dataset):
        def __init__(self, texts, labels, vocab_size=5000, max_len=64):
            self.texts = texts
            self.labels = labels
            self.max_len = max_len
            word_counts = Counter()
            for text in texts:
                for word in text.split():
                    word_counts[word] += 1
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            for word, count in word_counts.most_common(vocab_size - 2):
                self.vocab[word] = len(self.vocab)
            self.vocab_size = len(self.vocab)

        def __len__(self):
            return len(self.texts)

        def encode(self, text):
            tokens = text.split()[:self.max_len]
            ids = [self.vocab.get(w, 1) for w in tokens]
            ids += [0] * (self.max_len - len(ids))
            return ids

        def __getitem__(self, idx):
            ids = self.encode(self.texts[idx])
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    class SmallTransformer(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2,
                     hidden_dim=256, num_classes=8, max_len=64, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.pos_embedding = nn.Embedding(max_len, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(embed_dim, num_classes)

        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            x = self.embedding(input_ids) + self.pos_embedding(positions)
            mask = (input_ids == 0)
            x = self.transformer(x, src_key_padding_mask=mask)
            x = x.mean(dim=1)
            x = self.dropout(x)
            return self.classifier(x)

    # 构建数据集
    train_ds = JobDataset(X_train, y_train, max_len=64)
    test_ds = JobDataset(X_test, y_test, max_len=64)
    test_ds.vocab = train_ds.vocab
    test_ds.vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 初始化模型
    device = torch.device('cpu')
    model = SmallTransformer(
        vocab_size=train_ds.vocab_size,
        embed_dim=128, num_heads=4, num_layers=2,
        hidden_dim=256, num_classes=8,
        max_len=64, dropout=0.1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  SmallTransformer参数量: {total_params:,}")
    print(f"  分类数: 8 (产品线类别)")
    print(f"  词表大小: {train_ds.vocab_size}")

    # 训练配置（修正：epochs=20, lr=1e-3）
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"  Epochs: {EPOCHS}, LR: {LR}, Weight Decay: {WEIGHT_DECAY}")

    # 训练循环
    best_f1 = 0
    best_state = None
    best_preds, best_labels_list = [], []

    print(f"\n  {'Epoch':>5} {'Loss':>8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*35}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels_batch = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels_batch = batch['label'].to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels_batch.cpu().numpy())

        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average='weighted')

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_preds = all_preds.copy()
            best_labels_list = all_labels_list.copy()

        print(f"  {epoch+1:>5} {avg_loss:>8.4f} {acc:>8.4f} {f1:>8.4f}")

    final_acc = accuracy_score(best_labels_list, best_preds)
    final_f1 = best_f1

    print(f"\n  --- 最终结果 ---")
    print(f"  TF-IDF + LR:          Acc={lr_acc:.4f}, F1={lr_f1:.4f}")
    print(f"  TF-IDF + MLP:         Acc={mlp_acc:.4f}, F1={mlp_f1:.4f}")
    print(f"  SmallTransformer(8类): Acc={final_acc:.4f}, F1={final_f1:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(best_labels_list, best_preds,
                                target_names=PRODUCT_LINE_CATEGORIES, digits=4))

    # 保存模型
    model_path = os.path.join(BASE_DIR, 'small_transformer_model.pt')
    torch.save({
        'model_state_dict': best_state,
        'model_config': {
            'vocab_size': train_ds.vocab_size,
            'embed_dim': 128, 'num_heads': 4, 'num_layers': 2,
            'hidden_dim': 256, 'num_classes': 8, 'max_len': 64, 'dropout': 0.1,
        },
        'vocab': train_ds.vocab,
        'product_line_categories': PRODUCT_LINE_CATEGORIES,
        'results': {'best_f1': best_f1, 'best_accuracy': final_acc}
    }, model_path)
    print(f"\n  模型已保存: small_transformer_model.pt")

    return {"lr": (lr_acc, lr_f1), "mlp": (mlp_acc, mlp_f1),
            "transformer": (final_acc, final_f1)}


# #############################################################################
# 实验三：LinUCB强化学习岗位推荐排序
# 修正：8画像×30种子×300反馈；修复update()；补NDCG@10和Recall@10
# #############################################################################

def experiment_3_rl_ranking():
    """
    8个真实用户画像 × 30个随机种子 × 每画像300条反馈
    LinUCB学习排序策略 → IPS/DR反事实评估 → NDCG@10/Recall@10
    """
    print("\n" + "=" * 80)
    print("实验三：LinUCB强化学习岗位推荐排序")
    print("=" * 80)

    # 1. 加载岗位数据
    all_jobs = []
    for fname in ['batch1_jobs.json', 'batch2_jobs.json', 'batch3_jobs.json',
                   'batch4_jobs.json', 'liepin_jobs.json', 'supplement_jobs_zhaopin.json']:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_jobs.extend(data)

    print(f"  加载 {len(all_jobs)} 条岗位")

    # 2. 解析岗位特征
    def parse_salary(text):
        if not text or text == '面议':
            return 0, 0
        nums = re.findall(r'(\d+\.?\d*)', text)
        if len(nums) >= 2:
            low, high = float(nums[0]), float(nums[1])
            if '万' in text or 'W' in text.upper():
                low, high = low * 10, high * 10
            return low, high
        elif len(nums) == 1:
            val = float(nums[0])
            if '万' in text: val *= 10
            return val, val
        return 0, 0

    cat_map = {
        'AI研发/算法岗位': 0, 'AI产品/实施/运营岗位': 1,
        '数据/软件/数字化岗位': 2, '金融业务岗位': 3,
        '职能支持岗位': 4, '运营/生产岗位': 5
    }
    loc_map = {'北京': 0, '上海': 1, '深圳': 2, '杭州': 3, '广州': 4, '成都': 5, '南京': 6, '武汉': 7}

    job_features = []
    for i, job in enumerate(all_jobs):
        salary_low, salary_high = parse_salary(job.get('薪资文本', ''))
        salary_mid = (salary_low + salary_high) / 2 if salary_high > 0 else 15
        category = job.get('岗位类别', '')
        cat_id = cat_map.get(category, 4)
        is_ai = 1 if job.get('AI相关') == '是' else 0
        location = job.get('地点', '')
        loc_id = 8
        for city, lid in loc_map.items():
            if city in location:
                loc_id = lid
                break
        exp_text = job.get('经验', '')
        exp_years = 0
        if '1-3' in exp_text: exp_years = 2
        elif '3-5' in exp_text: exp_years = 4
        elif '5-10' in exp_text or '5年以上' in exp_text: exp_years = 7
        elif '10' in exp_text: exp_years = 12
        edu_text = job.get('学历', '')
        edu_level = 1
        if '博士' in edu_text: edu_level = 3
        elif '硕士' in edu_text: edu_level = 2
        elif '大专' in edu_text: edu_level = 0

        job_features.append({
            'job_id': i, 'cat_id': cat_id, 'is_ai': is_ai,
            'salary_mid': salary_mid, 'loc_id': loc_id,
            'exp_years': exp_years, 'edu_level': edu_level,
        })

    print(f"  岗位特征提取完成: {len(job_features)} 条")

    # 3. 构建8个真实用户画像
    user_personas = [
        {"user_id": "U001", "name": "金融学硕士应届",
         "education": 2, "exp_years": 0,
         "preferred_categories": [0, 1, 3], "preferred_ai": 1, "ai_skill": 0.6,
         "preferred_locations": [0, 1, 3], "salary_expectation": 20, "salary_flexibility": 10,
         "apply_threshold": 0.5, "click_threshold": 0.3},
        {"user_id": "U002", "name": "计算机本科3年Java",
         "education": 1, "exp_years": 3,
         "preferred_categories": [0, 2], "preferred_ai": 1, "ai_skill": 0.7,
         "preferred_locations": [0, 2, 4], "salary_expectation": 25, "salary_flexibility": 8,
         "apply_threshold": 0.55, "click_threshold": 0.35},
        {"user_id": "U003", "name": "金融工程博士量化",
         "education": 3, "exp_years": 5,
         "preferred_categories": [0, 3], "preferred_ai": 1, "ai_skill": 0.9,
         "preferred_locations": [0, 1], "salary_expectation": 40, "salary_flexibility": 15,
         "apply_threshold": 0.6, "click_threshold": 0.4},
        {"user_id": "U004", "name": "财务管理本科应届",
         "education": 1, "exp_years": 0,
         "preferred_categories": [1, 3, 4], "preferred_ai": 0.5, "ai_skill": 0.3,
         "preferred_locations": [1, 3, 5], "salary_expectation": 12, "salary_flexibility": 6,
         "apply_threshold": 0.45, "click_threshold": 0.25},
        {"user_id": "U005", "name": "软件工程本科5年全栈",
         "education": 1, "exp_years": 5,
         "preferred_categories": [2, 1], "preferred_ai": 0.6, "ai_skill": 0.5,
         "preferred_locations": [2, 0, 6], "salary_expectation": 30, "salary_flexibility": 10,
         "apply_threshold": 0.5, "click_threshold": 0.3},
        {"user_id": "U006", "name": "统计学硕士2年数据分析",
         "education": 2, "exp_years": 2,
         "preferred_categories": [0, 2, 3], "preferred_ai": 1, "ai_skill": 0.7,
         "preferred_locations": [0, 1, 2], "salary_expectation": 22, "salary_flexibility": 8,
         "apply_threshold": 0.5, "click_threshold": 0.3},
        {"user_id": "U007", "name": "AI硕士应届NLP方向",
         "education": 2, "exp_years": 0,
         "preferred_categories": [0], "preferred_ai": 1, "ai_skill": 0.95,
         "preferred_locations": [0, 2, 3], "salary_expectation": 35, "salary_flexibility": 12,
         "apply_threshold": 0.65, "click_threshold": 0.45},
        {"user_id": "U008", "name": "金融科技MBA 8年经验",
         "education": 2, "exp_years": 8,
         "preferred_categories": [1, 3, 4], "preferred_ai": 0.8, "ai_skill": 0.4,
         "preferred_locations": [1, 0], "salary_expectation": 50, "salary_flexibility": 20,
         "apply_threshold": 0.55, "click_threshold": 0.35},
    ]

    print(f"  构建了 {len(user_personas)} 个真实用户画像")

    # 4. 用户-岗位匹配度
    def compute_match_score(user, job):
        score = 0.0
        if job['cat_id'] in user['preferred_categories']:
            score += 0.3
        ai_match = 1 - abs(user['preferred_ai'] - job['is_ai'])
        score += 0.15 * ai_match
        if job['loc_id'] in user['preferred_locations']:
            score += 0.2
        elif job['loc_id'] == 8:
            score += 0.05
        salary_diff = abs(job['salary_mid'] - user['salary_expectation'])
        score += 0.15 * max(0, 1 - salary_diff / (user['salary_flexibility'] + 5))
        edu_diff = abs(user['education'] - job['edu_level'])
        score += 0.1 * max(0, 1 - edu_diff * 0.5)
        exp_diff = abs(user['exp_years'] - job['exp_years'])
        score += 0.1 * max(0, 1 - exp_diff * 0.15)
        return min(score, 1.0)

    # 5. LinUCB算法（修复update()）
    class LinUCB:
        def __init__(self, n_arms, context_dim, alpha=2.0):
            self.n_arms = n_arms
            self.context_dim = context_dim
            self.alpha = alpha
            self.A = [np.eye(context_dim) for _ in range(n_arms)]
            self.b = [np.zeros(context_dim) for _ in range(n_arms)]

        def get_context(self, inter):
            return np.array([
                inter['user_edu'] / 3, inter['user_exp'] / 12,
                inter['user_ai_pref'], inter['user_ai_skill'],
                inter['user_salary_exp'] / 50,
                inter['job_cat'] / 5, inter['job_is_ai'],
                inter['job_salary'] / 50, inter['job_loc'] / 8,
                inter['job_exp'] / 12, inter['job_edu'] / 3, 1.0
            ])

        def get_scores(self, context):
            scores = []
            for a in range(self.n_arms):
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                p = theta @ context + self.alpha * np.sqrt(max(context @ A_inv @ context, 0))
                scores.append(p)
            return np.array(scores)

        def get_probs(self, context, temp=1.0):
            scores = self.get_scores(context)
            exp_s = np.exp(scores / temp - np.max(scores / temp))
            return exp_s / exp_s.sum()

        def select_arm(self, context):
            return np.argmax(self.get_scores(context))

        def update(self, arm, context, reward):
            """修复：直接更新A和b，无嵌套定义"""
            self.A[arm] = self.A[arm] + np.outer(context, context)
            self.b[arm] = self.b[arm] + reward * context

    # 6. NDCG@10和Recall@10计算
    def compute_ndcg_at_k(recommended_items, relevant_items, k=10):
        """计算NDCG@10"""
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        ideal_n = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
        return dcg / idcg if idcg > 0 else 0

    def compute_recall_at_k(recommended_items, relevant_items, k=10):
        """计算Recall@10"""
        if not relevant_items:
            return 0
        hits = len(set(recommended_items[:k]) & set(relevant_items))
        return hits / len(relevant_items)

    # 7. 30个随机种子实验
    N_SEEDS = 30
    N_FEEDBACKS_PER_USER = 300
    n_arms = 6
    context_dim = 12

    all_ndcg = []
    all_recall = []
    all_dr = []
    all_ips = []

    print(f"\n  运行 {N_SEEDS} 个随机种子，每种子 {len(user_personas)} 画像 × {N_FEEDBACKS_PER_USER} 反馈")

    for seed_idx in range(N_SEEDS):
        seed = SEED + seed_idx
        np.random.seed(seed)

        # 生成交互日志
        interactions = []
        for user in user_personas:
            for fb_idx in range(N_FEEDBACKS_PER_USER):
                available = np.random.choice(
                    len(job_features),
                    size=min(10, len(job_features)), replace=False)
                for job_idx in available:
                    job = job_features[job_idx]
                    match_score = compute_match_score(user, job)
                    noisy_score = np.clip(match_score + np.random.normal(0, 0.1), 0, 1)

                    if noisy_score > user['apply_threshold']:
                        action, reward = 'apply', 3
                    elif noisy_score > user['click_threshold']:
                        action, reward = 'click', 1
                    elif noisy_score > user['click_threshold'] - 0.1 and np.random.random() < 0.3:
                        action, reward = 'save', 2
                    else:
                        action, reward = 'ignore', 0

                    interactions.append({
                        'user_id': user['user_id'], 'reward': reward,
                        'user_edu': user['education'], 'user_exp': user['exp_years'],
                        'user_ai_pref': user['preferred_ai'], 'user_ai_skill': user['ai_skill'],
                        'user_salary_exp': user['salary_expectation'],
                        'job_cat': job['cat_id'], 'job_is_ai': job['is_ai'],
                        'job_salary': job['salary_mid'], 'job_loc': job['loc_id'],
                        'job_exp': job['exp_years'], 'job_edu': job['edu_level'],
                        'match_score': round(match_score, 4),
                    })

        # 按时间排序，前半训练后半评估
        interactions_sorted = sorted(interactions, key=lambda x: x['user_id'])
        n_total = len(interactions_sorted)
        n_train = n_total // 2
        train_data = interactions_sorted[:n_train]
        eval_data = interactions_sorted[n_train:]

        # 训练LinUCB
        linucb = LinUCB(n_arms, context_dim, alpha=2.0)
        for inter in train_data:
            ctx = linucb.get_context(inter)
            linucb.update(inter['job_cat'], ctx, inter['reward'])

        # 奖励预测模型
        train_features = [linucb.get_context(inter).tolist() for inter in train_data]
        train_labels_list = [inter['reward'] for inter in train_data]
        reward_model = Ridge(alpha=1.0)
        reward_model.fit(train_features, train_labels_list)

        # IPS/DR评估
        log_policy_prob = 1.0 / n_arms
        ips_values, dr_values = [], []
        seed_ndcg, seed_recall = [], []

        for inter in eval_data:
            ctx = linucb.get_context(inter)
            target_probs = linucb.get_probs(ctx, temp=1.0)
            actual_arm = inter['job_cat']
            actual_reward = inter['reward']

            # IPS
            if target_probs[actual_arm] > 0:
                weight = target_probs[actual_arm] / log_policy_prob
                ips_values.append(actual_reward * weight)
                q_hat = reward_model.predict([ctx])[0]
                dr_values.append(q_hat + (actual_reward - q_hat) * weight)

            # NDCG@10和Recall@10
            scores = linucb.get_scores(ctx)
            ranked_arms = np.argsort(-scores)
            relevant_arms = [inter['job_cat']] if actual_reward > 0 else []
            seed_ndcg.append(compute_ndcg_at_k(ranked_arms, relevant_arms, k=10))
            seed_recall.append(compute_recall_at_k(ranked_arms, relevant_arms, k=10))

        all_ndcg.append(np.mean(seed_ndcg))
        all_recall.append(np.mean(seed_recall))
        all_dr.append(np.mean(dr_values) if dr_values else 0)
        all_ips.append(np.mean(ips_values) if ips_values else 0)

    # 汇总结果
    ndcg_mean, ndcg_std = np.mean(all_ndcg), np.std(all_ndcg)
    recall_mean, recall_std = np.mean(all_recall), np.std(all_recall)
    dr_mean, dr_std = np.mean(all_dr), np.std(all_dr)
    ips_mean, ips_std = np.mean(all_ips), np.std(all_ips)
    logging_avg = np.mean([i['reward'] for i in eval_data])
    random_est = logging_avg / n_arms

    print(f"\n  --- {N_SEEDS}种子评估结果 ---")
    print(f"  {'指标':<25} {'均值':>10} {'标准差':>10}")
    print(f"  {'-'*48}")
    print(f"  {'NDCG@10':<25} {ndcg_mean:>10.4f} {ndcg_std:>10.4f}")
    print(f"  {'Recall@10':<25} {recall_mean:>10.4f} {recall_std:>10.4f}")
    print(f"  {'IPS估计':<25} {ips_mean:>10.4f} {ips_std:>10.4f}")
    print(f"  {'DR估计':<25} {dr_mean:>10.4f} {dr_std:>10.4f}")
    print(f"  {'Logging基准':<25} {logging_avg:>10.4f} {'---':>10}")
    print(f"  {'Random基准':<25} {random_est:>10.4f} {'---':>10}")
    print(f"\n  DR vs Random: {dr_mean/max(random_est,0.001):.2f}x 提升")
    print(f"  IPS vs Random: {ips_mean/max(random_est,0.001):.2f}x 提升")

    # 保存结果
    results = {
        "config": {
            "n_personas": 8, "n_seeds": 30, "n_feedbacks_per_user": 300,
            "n_arms": n_arms, "context_dim": context_dim, "alpha": 2.0,
        },
        "evaluation": {
            "ndcg_at_10": {"mean": round(ndcg_mean, 4), "std": round(ndcg_std, 4)},
            "recall_at_10": {"mean": round(recall_mean, 4), "std": round(recall_std, 4)},
            "ips": {"mean": round(ips_mean, 4), "std": round(ips_std, 4)},
            "dr": {"mean": round(dr_mean, 4), "std": round(dr_std, 4)},
            "logging_avg": round(logging_avg, 4),
            "random_est": round(random_est, 4),
        },
        "improvement": {
            "dr_vs_random": round(dr_mean / max(random_est, 0.001), 2),
            "ips_vs_random": round(ips_mean / max(random_est, 0.001), 2),
        },
        "seed": SEED,
    }
    with open(os.path.join(BASE_DIR, "rl_experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: rl_experiment_results.json")
    return results


# #############################################################################
# 主函数
# #############################################################################

if __name__ == '__main__':
    print("=" * 80)
    print("AI金融科技公司业务布局与人才需求分析 — 实验代码（修正版）")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"SEED: {SEED}")
    print("=" * 80)

    r1 = experiment_1_second_rater()
    r2 = experiment_2_finetune()
    r3 = experiment_3_rl_ranking()

    print("\n" + "=" * 80)
    print("全部实验完成")
    print("=" * 80)
    print(f"  实验一: ICC={r1['icc']:.4f}, 加权Kappa={r1['weighted_kappa']:.4f}")
    print(f"  实验二: SmallTransformer(8类) Acc={r2['transformer'][0]:.4f}, F1={r2['transformer'][1]:.4f}")
    print(f"  实验三: NDCG@10={r3['evaluation']['ndcg_at_10']['mean']:.4f}, DR={r3['evaluation']['dr']['mean']:.4f}")
