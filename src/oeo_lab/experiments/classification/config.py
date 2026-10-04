# -*- coding: utf-8 -*-
"""全局配置:特征列、实验网格、行为开关。集中管理便于复现与调参。"""
import os

# 要分箱的连续特征列(与原 BitcoinAddress2Context 一致)
COL_NAMES = ["avgUseGap", "label_cnt", "balance", "total_received", "received_cnt",
             "avg_receive", "total_sent", "sent_cnt", "avg_sent"]

# ⭐ 新增：无泄漏特征集
COL_NAMES_NO_LEAK = [c for c in COL_NAMES if c != 'label_cnt']

COL_NAMES_LEAK = COL_NAMES


# 建议添加标记
LEAKED_FEATURES = ['label_cnt']  # 明确标记泄漏特征

USE_LEAKED_FEATURES = False   # 唯一开关：True=含label_cnt(历史)，False=无泄漏(新)
COL_NAMES = COL_NAMES_LEAK if USE_LEAKED_FEATURES else COL_NAMES_NO_LEAK

# 标签来源: label = cate_id - 1, 仅保留 label < N_CLASS 的类别
LABEL_FROM = "cate_id"
LABEL_OFFSET = 1          # label = cate_id - LABEL_OFFSET
N_CLASS = 4               # 仅保留 4 个行业类别

# 实验网格
DEFAULT_MAX_BINS = [250, 300, 350, 400, 450, 500]
DEFAULT_CV_SEEDS = [11, 37, 53, 79, 97, 101]   # 多种子 -> 学习曲线方差/显著性
DEFAULT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_FOLD = 5
RANDOM_SEED = 42

# 行为开关
DEDUP_TRAIN = True            # 划分后仅对训练集去重(X/y 同步), 测试集绝不去重
INCLUDE_RAW_BASELINES = True  # 额外加入"原始连续特征"对照组(树模型不分箱)
RESULTS_ROOT = "./results"


def default_dataset_csv():
    from pathlib import Path
    root = Path(os.environ.get("OEO_DATA_ROOT", Path(__file__).resolve().parents[4] / "data"))
    return str(root / "raw" / "bitcoin" / "getFeatures.csv")


# ============ 输出文件命名集中管理（唯一真源） ============
FC_DIR_PREFIX = "formalContext_"     # 每个 bin 的子目录前缀
RESULT_STEM   = "综合对比实验"        # 每个 frac 的明细结果词干
SUMMARY_STEM  = "new综合对比实验"     # 总汇总词干（后面接 {sfx}_总汇总）
STATS_DIRNAME = "_statistical_analysis"
PARTIAL_STEM  = ".partial"           # sidecar 目录前缀

def proto_suffix(protocol):
    """协议A(address)空后缀保持历史命名; 其余协议插入 _{protocol}。"""
    return "" if protocol == "address" else f"_{protocol}"

def fc_dir(results_root, max_bin):
    return f"{results_root}/{FC_DIR_PREFIX}{max_bin}"

def result_file(parent_dir, frac, protocol):
    return f"{parent_dir}/{RESULT_STEM}{proto_suffix(protocol)}_frac_{frac}.xlsx"

def result_tmp(parent_dir, frac, protocol):
    return f"{parent_dir}/.tmp{proto_suffix(protocol)}_frac_{frac}.xlsx"

def summary_file(parent_dir, protocol):
    return f"{parent_dir}/{SUMMARY_STEM}{proto_suffix(protocol)}_总汇总.xlsx"

def partial_dir(parent_dir, protocol):
    return f"{parent_dir}/{PARTIAL_STEM}{proto_suffix(protocol)}"

def sidecar_file(pdir, frac, model_name, seed):
    return f"{pdir}/frac{frac}_{model_name}_seed{seed}.pkl"

def stats_dir(results_root):
    return f"{results_root}/{STATS_DIRNAME}"

def stats_file(out_dir, frac, group_tag, protocol, metric):
    return f"{out_dir}/frac_{frac}_{group_tag}{proto_suffix(protocol)}_显著性_{metric}.xlsx"