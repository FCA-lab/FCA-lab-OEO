# -*- coding: utf-8 -*-
"""实验入口。

运行:
    python -m oeo_lab.experiments.classification.run                      # 用 config 默认值(协议A)
    python -m oeo_lab.experiments.classification.run --protocol entity    # 协议B: 实体级分组切分(无泄漏)
    python -m oeo_lab.experiments.classification.run --no-dedup           # 关闭训练集去重
    python -m oeo_lab.experiments.classification.run --no-raw             # 关闭原始特征对照组
    python -m oeo_lab.experiments.classification.run --data /path/getFeatures.csv --n-jobs 16
    python -m oeo_lab.experiments.classification.run --n-jobs 12 --tasks-per-worker 1   # 每任务重建进程, 内存峰值最低
"""
import argparse

from . import config
from .runner import run_all_experiments_parallel

import warnings

warnings.filterwarnings("ignore", message=".*n_jobs.*has no effect.*", category=FutureWarning)

def parse_args():
    ap = argparse.ArgumentParser(description="比特币地址分类 无泄漏并行实验")
    ap.add_argument("--data", default=config.default_dataset_csv(), help="原始特征 CSV(分箱前)")
    ap.add_argument("--results-root", default=config.RESULTS_ROOT)
    ap.add_argument("--max-bins", type=int, nargs="*", default=config.DEFAULT_MAX_BINS)
    ap.add_argument("--fracs", type=float, nargs="*", default=config.DEFAULT_FRACS)
    ap.add_argument("--seeds", type=int, nargs="*", default=config.DEFAULT_CV_SEEDS)
    ap.add_argument("--fold", type=int, default=config.DEFAULT_FOLD)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--metric", default="宏平均F1值_mean", help="显著性分析所用指标")
    ap.add_argument("--protocol", choices=["address", "entity"], default="address",
                    help="address=协议A地址级随机切分(原命名); "
                         "entity=协议B实体级分组切分(文件名带_entity, 无实体泄漏)")
    ap.add_argument("--tasks-per-worker", type=int, default=1,
                    help="每个 worker 进程被回收前最多跑几个任务(chunk=有效核数×该值)。"
                         "=1 内存峰值最低、每任务重建进程(推荐); 调大用内存换吞吐。")
    ap.add_argument("--keep-partial", action="store_true",
                    help="聚合成 xlsx 后保留 .partial sidecar(调试用); 默认聚合成功即清理。")
    ap.add_argument("--no-dedup", action="store_true", help="关闭训练集去重")
    ap.add_argument("--no-resume", action="store_true", help="不续跑, 全部重算")
    ap.add_argument("--no-raw", action="store_true", help="关闭原始连续特征对照组")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="内存自适应阈值(GB): 补位新 worker 前若系统可用内存 < 该值, "
                         "本轮不新增, 等下一个任务完成释放后再探测。=0 关闭。")
    ap.add_argument("--worker-ramp-sec", type=float, default=3.0,
                    help="每提交一个新 worker 后等待秒数, 让其开始占内存使下次读数更准。=0 关闭。")
    return ap.parse_args()


def main():
    args = parse_args()
    config.INCLUDE_RAW_BASELINES = not args.no_raw      # models.build_models 默认 include_raw=True
    run_all_experiments_parallel(
        dataset_csv=args.data, max_bins=args.max_bins, cv_random_seeds=args.seeds,
        frac_list=args.fracs, results_root=args.results_root, random_seed=config.RANDOM_SEED,
        fold=args.fold, n_jobs=args.n_jobs, resume=not args.no_resume,
        dedup_train=not args.no_dedup,
        protocol=args.protocol,
        tasks_per_worker=args.tasks_per_worker,          # ← 新增
        keep_partial=args.keep_partial,                  # ← 新增
        min_free_gb=args.min_free_gb,                    # ← 新增
        worker_ramp_sec=args.worker_ramp_sec)            # ← 新增



if __name__ == "__main__":
    main()