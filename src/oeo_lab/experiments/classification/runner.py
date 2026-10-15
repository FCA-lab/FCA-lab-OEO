# -*- coding: utf-8 -*-
"""并行调度(细粒度) + 子任务级 sidecar 落盘 + 断点续跑 + worker 逐任务回收 + 内存自适应补位。

内存与并行(核心改动):
    旧版 joblib.Parallel(n_jobs=N) 复用常驻 worker, 任务结束 != 进程退出, RES 累积 -> OOM;
    "分 chunk、chunk 末 shutdown" 又引入屏障长尾(快任务完成后空转等整批最慢任务)。
    本版用标准库 ProcessPoolExecutor(max_tasks_per_child=K)(Python≥3.11 原生):
      - worker 跑满 K 个任务即退出、内存 100% 归还 OS; 池自动补新 worker;
      - 指定 max_tasks_per_child 且未传 mp_context 时默认 spawn(子进程全新解释器, 无继承泄漏)。
    并叠加"内存自适应受控提交": 始终只保持 ≤ n_jobs 个在飞任务; 每当有任务完成需补位时,
    先探测系统可用内存(Linux MemAvailable), 若 < min_free_gb 则本轮不新增 worker, 等下一个
    任务完成、内存释放后再探测补位。补位数永不超过 n_jobs。

断点粒度:
    每个 (bin,frac,model,seed) 子任务算完立即原子写 sidecar .pkl, resume 靠文件存在性跳过,
    崩溃最多损失"飞行中"的 ≤ n_jobs 个。全跑完从 sidecar 聚合成与旧版逐字一致的 xlsx。

协议与文件命名(不变):
    protocol='address' (协议A): 综合对比实验_frac_{frac}.xlsx / new综合对比实验_总汇总.xlsx
    protocol='entity'  (协议B): 综合对比实验_entity_frac_{frac}.xlsx / new综合对比实验_entity_总汇总.xlsx
"""
import os
import gc
import time
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd

from .baselines import build_models, HAS_RC, HAS_XGB, HAS_LGBM
from .evaluate import run_one_seed, aggregate_model


def _proto_suffix(protocol):
    """协议A保持历史命名(空后缀); 其余协议插入 _{protocol} 后缀。"""
    return "" if protocol == "address" else f"_{protocol}"


def _effective_jobs(n_jobs):
    """把 joblib 的 n_jobs 语义(-1=全核, -2=全核-1, ...)折算成正整数, 供 max_workers。"""
    cpu = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return cpu
    if n_jobs < 0:
        return max(1, cpu + 1 + n_jobs)
    return n_jobs


def _avail_mem_bytes():
    """系统当前可用内存(字节)。Linux 优先 /proc/meminfo:MemAvailable(可回收缓存计入, 最准);
    退回 psutil; 都不可用则返回 None(视为不限制)。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024      # kB -> B
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


# ----------------------------------------------------------------------------
# sidecar: 子任务级持久化
# ----------------------------------------------------------------------------
def _partial_dir(parent_dir, protocol):
    d = os.path.join(parent_dir, f".partial{_proto_suffix(protocol)}")
    os.makedirs(d, exist_ok=True)
    return d


def _sidecar_path(pdir, frac, model_name, seed):
    return os.path.join(pdir, f"frac{frac}_{model_name}_seed{seed}.pkl")


def _run_and_dump(dataset_csv, max_bin, frac, model_name, seed, fold,
                  random_seed, dedup_train, protocol, sidecar):
    """在 worker 内执行单个子任务, 结果原子落盘为 sidecar; 只回传轻量状态, 不回传大对象。"""
    r = run_one_seed(dataset_csv, max_bin, frac, model_name, seed,
                     fold, random_seed, dedup_train, protocol)
    tmp = sidecar + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, sidecar)          # 原子写: 崩溃不会留半截文件污染 resume
    gc.collect()
    return (max_bin, frac, model_name, seed)


# ----------------------------------------------------------------------------
# 聚合落盘(与旧版逐字一致)
# ----------------------------------------------------------------------------
def _write_frac_file(parent_dir, max_bin, frac, agg_list, order, protocol):
    sfx = _proto_suffix(protocol)
    agg_list.sort(key=lambda r: order.get(r["model_name"], 0))
    fold_rows, summary_rows, confusion_rows, report_rows = [], [], [], []
    for r in agg_list:
        fold_rows += r["fold_rows"]
        summary_rows.append(r["summary_row"])
        confusion_rows += r["confusion_rows"]
        report_rows += r["report_rows"]
    fn = f"{parent_dir}/综合对比实验{sfx}_frac_{frac}.xlsx"
    tmp = f"{parent_dir}/.tmp{sfx}_frac_{frac}.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        pd.DataFrame(fold_rows).to_excel(w, sheet_name="每折指标", index=False)
        pd.DataFrame(summary_rows).to_excel(w, sheet_name="汇总指标", index=False)
        pd.DataFrame(confusion_rows).to_excel(w, sheet_name="混淆矩阵汇总", index=False)
        pd.DataFrame(report_rows).to_excel(w, sheet_name="分类报告", index=False)
    os.replace(tmp, fn)
    print(f"    已保存: {fn}")


def _aggregate_frac_from_sidecars(parent_dir, max_bin, frac, model_names,
                                  cv_random_seeds, order, protocol,
                                  n_models, n_seeds, fold, keep_partial=False):
    """把某 (max_bin, frac) 的所有 sidecar 聚合成 xlsx。
    仅当全部 n_models×n_seeds 子任务到齐才写盘并清理; 否则保留 sidecar 等待 resume 补齐。
    """
    pdir = _partial_dir(parent_dir, protocol)
    per_model = defaultdict(list)
    present = 0
    for mn in model_names:
        for sd in cv_random_seeds:
            sc = _sidecar_path(pdir, frac, mn, sd)
            if os.path.exists(sc):
                with open(sc, "rb") as f:
                    per_model[mn].append(pickle.load(f))
                present += 1
    expected = n_models * n_seeds
    if present < expected:
        print(f"    [跳过] max_bin={max_bin} frac={frac} 仅 {present}/{expected} 子任务到齐, "
              f"保留 sidecar 待下次 resume 补齐 (不写 xlsx)")
        return

    agg_list = [aggregate_model(srs, n_seeds, fold) for mn, srs in per_model.items() if srs]
    _write_frac_file(parent_dir, max_bin, frac, agg_list, order, protocol)

    if not keep_partial:                              # 成功写盘后清理 sidecar
        for mn in model_names:
            for sd in cv_random_seeds:
                sc = _sidecar_path(pdir, frac, mn, sd)
                if os.path.exists(sc):
                    os.remove(sc)


def _refresh_total_summary(parent_dir, frac_list, frac_path, protocol):
    sfx = _proto_suffix(protocol)
    frames = []
    for fr in frac_list:
        fn = frac_path(fr)
        if os.path.exists(fn):
            try:
                frames.append(pd.read_excel(fn, sheet_name="汇总指标"))
            except Exception as exc:
                print(f"    跳过损坏文件 {fn}: {exc}")
    if frames:
        out = f"{parent_dir}/new综合对比实验{sfx}_总汇总.xlsx"
        pd.concat(frames, ignore_index=True).to_excel(out, index=False)
        print(f"    总汇总已刷新: {out}")


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def run_all_experiments_parallel(dataset_csv, max_bins, cv_random_seeds, frac_list,
                                 results_root="./results", random_seed=42, fold=5,
                                 n_jobs=-1, resume=True, dedup_train=True,
                                 protocol="address", tasks_per_worker=1,
                                 keep_partial=False, min_free_gb=2.0,
                                 worker_ramp_sec=3.0):
    """
    tasks_per_worker: 传给 max_tasks_per_child。worker 跑满该任务数即退出、内存归还 OS。
                      =1 每任务重建进程, 内存峰值最低, 推荐。
    min_free_gb:      内存自适应阈值(GB)。补位新 worker 前若系统可用内存 < 该值, 本轮不新增,
                      等下一个任务完成、内存释放后再探测。=0 关闭该保护。
    worker_ramp_sec:  每提交一个新 worker 后等待的秒数, 让其开始分配内存, 使下次内存读数更准
                      (spawn 下新进程 import+建矩阵需时间, 否则会瞬时过量提交)。=0 关闭。
    keep_partial:     True 则聚合后保留 sidecar(调试用); 默认聚合成功即清理。
    """
    sfx = _proto_suffix(protocol)
    model_names = list(build_models(random_seed).keys())
    order = {m: i for i, m in enumerate(model_names)}
    n_models = len(model_names)
    n_seeds = len(cv_random_seeds)
    eff_jobs = _effective_jobs(n_jobs)
    min_free = int(min_free_gb * (1024 ** 3))

    print(f"原始数据: {dataset_csv}")
    print(f"切分协议: {protocol} ({'地址级随机切分' if protocol == 'address' else '实体级分组切分(无泄漏)'})")
    print(f"模型({n_models}): {model_names}")
    print(f"强基线: XGBoost={HAS_XGB} LightGBM={HAS_LGBM} | 本文方法 rc_3woc={HAS_RC} | 训练集去重={dedup_train}")
    print(f"并行: max_workers={eff_jobs} | max_tasks_per_child={max(1, tasks_per_worker)} "
          f"| 内存阈值={min_free_gb}GB | ramp={worker_ramp_sec}s | 逐任务回收+内存自适应补位")

    # ==== 1. 确定待跑 (max_bin, frac): 最终 xlsx 已存在的整体跳过 ====
    parent_of = {}
    pending = []
    for max_bin in max_bins:
        parent_dir = f"{results_root}/formalContext_{max_bin}"
        os.makedirs(parent_dir, exist_ok=True)
        parent_of[max_bin] = parent_dir
        for fr in frac_list:
            fn = f"{parent_dir}/综合对比实验{sfx}_frac_{fr}.xlsx"
            if resume and os.path.exists(fn):
                continue
            pending.append((max_bin, fr))

    if not pending:
        print("全部 (max_bin, frac) 均已完成, 仅刷新总汇总。")
    else:
        pending.sort(key=lambda t: (-t[0], -t[1]))   # 大 bin、大 frac(最慢)先提交

        # ==== 2. 展平到子任务, 跳过已落 sidecar 的(细粒度 resume) ====
        all_tasks = [(mb, fr, mn, sd) for (mb, fr) in pending
                     for mn in model_names for sd in cv_random_seeds]
        to_run = []
        for (mb, fr, mn, sd) in all_tasks:
            sc = _sidecar_path(_partial_dir(parent_of[mb], protocol), fr, mn, sd)
            if resume and os.path.exists(sc):
                continue
            to_run.append((mb, fr, mn, sd, sc))
        n_cached = len(all_tasks) - len(to_run)
        print(f"待跑组合: {len(pending)} 个 (max_bin, frac) | 子任务总数 {len(all_tasks)} "
              f"(已落 sidecar 跳过 {n_cached}, 实际待跑 {len(to_run)})")

        # ==== 3. 阶段一: 受控提交 —— 逐任务回收 + 内存自适应补位, 补位数永不超过 eff_jobs ====
        if to_run:
            tasks_iter = iter(to_run)
            in_flight = {}                 # future -> (mb, fr, mn, sd)
            n_total = len(to_run)
            n_done = 0
            exhausted = False
            blocked_notified = False

            with ProcessPoolExecutor(max_workers=eff_jobs,
                                     max_tasks_per_child=max(1, tasks_per_worker)) as executor:
                while in_flight or not exhausted:
                    # 补位: 不超过 eff_jobs, 且仅当可用内存 >= 阈值才新增 worker
                    while not exhausted and len(in_flight) < eff_jobs:
                        av = _avail_mem_bytes()
                        if min_free > 0 and av is not None and av < min_free:
                            if not blocked_notified:
                                print(f"[内存保护] 可用 {av/1024**3:.1f}GB < {min_free_gb}GB, "
                                      f"本轮暂不新增 worker(当前活跃 {len(in_flight)}), 等任务释放…")
                                blocked_notified = True
                            break
                        try:
                            mb, fr, mn, sd, sc = next(tasks_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        fut = executor.submit(_run_and_dump, dataset_csv, mb, fr, mn, sd,
                                              fold, random_seed, dedup_train, protocol, sc)
                        in_flight[fut] = (mb, fr, mn, sd)
                        blocked_notified = False
                        if worker_ramp_sec > 0:
                            time.sleep(worker_ramp_sec)   # 给新 worker 时间分配内存, 使下次读数更准

                    if not in_flight:
                        if exhausted:
                            break
                        # 任务未完但被内存卡住且当前无活跃 worker: 稍等重试, 避免忙等
                        time.sleep(max(1.0, worker_ramp_sec))
                        continue

                    # 等至少一个任务完成 -> 释放内存 -> 回到顶部重新探测补位
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        mb, fr, mn, sd = in_flight.pop(fut)
                        try:
                            fut.result()          # sidecar 已在 worker 内落盘; 这里只探测异常
                        except Exception as exc:
                            print(f"[warn] 子任务失败 bin={mb} frac={fr} {mn} seed{sd}: {exc} "
                                  f"(无 sidecar, 下次 resume 会重跑)")
                        n_done += 1
                        if n_done % eff_jobs == 0 or n_done == n_total:
                            print(f"[进度] {n_done}/{n_total} 子任务完成 "
                                  f"(活跃 worker {len(in_flight)}/{eff_jobs})")
            gc.collect()   # with 退出即 shutdown(wait=True), 全部 worker 已关闭

        # ==== 4. 阶段二: 从 sidecar 聚合成 xlsx(轻量, 主进程完成) ====
        for (mb, fr) in pending:
            _aggregate_frac_from_sidecars(parent_of[mb], mb, fr, model_names,
                                          cv_random_seeds, order, protocol,
                                          n_models, n_seeds, fold, keep_partial=keep_partial)

    # ==== 5. 各 bin 刷新总汇总 ====
    for max_bin in max_bins:
        parent_dir = parent_of[max_bin]
        frac_path = lambda fr: f"{parent_dir}/综合对比实验{sfx}_frac_{fr}.xlsx"
        _refresh_total_summary(parent_dir, frac_list, frac_path, protocol)
    print("\n全部实验结束！")


# ===== 新分类实验接口：任务计算与报表保存分离，供 UCI/后续 Bitcoin 共用 =====
def classification_worker(task):
    """工作进程入口；控制台即时显示数据集、模型、分箱和折号。"""
    from .evaluate import evaluate_fold
    from threadpoolctl import threadpool_limits
    label = (f"{task['name']} | {task.get('protocol','stratified')} | {task['model_name']} | train={task.get('train_budget',len(task['split']['train_indices']))} | fraction={task.get('fraction','n/a')} | bins={task['max_bins']} | "
             f"seed={task['split']['seed']} fold={task['split']['fold']}")
    print(f'[开始] {label} | PID={os.getpid()}', flush=True)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            result = evaluate_fold(task)
        print(f'[完成计算] {label} | {time.perf_counter()-started:.2f}s', flush=True)
        return result
    except Exception as exc:
        print(f'[失败] {label} | {type(exc).__name__}: {exc}', flush=True)
        raise


# 2026-10-06：避免全量 Bitcoin 规则记录超过 Excel 单表行数而导出失败。
def write_rule_sheets(writer, records, rows_per_sheet=1_000_000):
    """按行数拆分逐规则表，避免超过 Excel 单表上限；检查点仍保留全量。"""
    if not 1 <= rows_per_sheet <= 1_048_575:
        raise ValueError('Invalid Excel data row limit')
    for start in range(0, max(1,len(records)), rows_per_sheet):
        suffix = '' if start == 0 else '_' + str(start//rows_per_sheet+1)
        pd.DataFrame(records[start:start+rows_per_sheet]).to_excel(
            writer, sheet_name='rules'+suffix, index=False)


def summarize_classification(tasks, directory, output, template):
    """汇总折结果；先按种子合并 OOF 计数，再对不同种子的指标求均值/标准差。"""
    import json
    from pathlib import Path
    from ..common.checkpoints import same_task
    folds, states, rules = [], [], []
    for task in tasks:
        path = directory/template.format(task_id=task['id'])
        saved = json.loads(path.read_text()) if path.exists() else {}
        # 与恢复检查一致：仅调度参数不同的历史成功任务仍可汇总。
        if not same_task(saved.get('task'), task): saved = {}
        base = {'task_id':task['id'], 'dataset':task['name'], 'model':task['model_name'],
                'max_bins':task['max_bins'], 'seed':task['split']['seed'], 'fold':task['split']['fold']}
        if 'protocol' in task:
            base.update(protocol=task['protocol'], entity_overlap=task['split']['entity_overlap'])
        if 'fraction' in task:
            base.update(fraction=task['fraction'], n_train_entities=task['n_train_entities'])
        # 独立样本预算必须进入记录及聚合键，不能混合不同训练规模。
        if 'train_budget' in task:
            base.update(train_budget=task['train_budget'], train_per_class=task['train_per_class'])
        for key in ('feature_set','train_end','test_start','test_end'):
            if key in task:base[key]=task[key]
        status = saved.get('status','pending')
        states.append({**base,'status':status,'error':saved.get('error')})
        variants = task['voting'] if task['model']['estimator']=='oeo' else ['baseline']
        if status != 'ok':
            if 'prediction_variants' in task:
                folds.extend({**base,'voting':v['voting'],'rule_group':v['rule_group'],
                              'study':study,'status':status}
                             for v in task['prediction_variants'] for study in v['studies'])
            else:
                folds.extend({**base,'voting':v,'status':status} for v in variants)
            continue
        result = saved['result']
        timing = {k:result[k] for k in ['fit_seconds','preprocessing_fit_seconds',
            'preprocessing_test_seconds','train_seconds','n_train','attributes','rule_count','zero_support_rules']}
        for row in result['results']:
            folds.append({**base,**timing,**row,'status':'ok'})
        for rule in result['rules']:
            rules.append({**base,**rule})
    frame = pd.DataFrame(folds)
    per_seed = []
    keys = ['dataset','model','max_bins','voting','seed']
    if 'study' in frame:
        keys = ['study','rule_group'] + keys
    if 'protocol' in frame:
        keys.insert(1, 'protocol')  # 两种划分分别聚合，禁止混合平均。
    if 'fraction' in frame:
        keys.insert(2, 'fraction')  # 各比例独立聚合固定测试折的 OOF 结果。
    if 'train_budget' in frame:
        keys = ['train_budget','train_per_class'] + keys
    for field in ('feature_set','train_end','test_start','test_end'):
        if field in frame:keys.insert(0,field)
    for group_key, group in frame.groupby(keys,dropna=False,sort=False):
        row = dict(zip(keys,group_key))
        row.update(expected_folds=len(group), completed_folds=int(group.status.eq('ok').sum()))
        row['complete'] = row['completed_folds']==row['expected_folds']
        if row['complete']:
            total=int(group.n_test.sum()); accepted=int(group.n_classified.sum()); correct=int(group.n_correct.sum())
            # 每个种子内，各测试折不重叠；累计 TP/FP/FN 得到完整 OOF 宏 F1。
            per_class = {}
            for values in group.per_class:
                for c in values:
                    a=per_class.setdefault(c['label'],[0,0,0])
                    for i,k in enumerate(['tp','fp','fn']): a[i]+=c[k]
            f1=[2*t/(2*t+p+n) if 2*t+p+n else 0.0 for t,p,n in per_class.values()]
            row.update(n_test=total,n_classified=accepted,n_correct=correct,
                       coverage=accepted/total,classified_accuracy=correct/accepted if accepted else None,
                       overall_correct_rate=correct/total,macro_f1=sum(f1)/len(f1))
            row['macro_precision']=sum(t/(t+p) if t+p else 0.0 for t,p,n in per_class.values())/len(per_class)
            row['macro_recall']=sum(t/(t+n) if t+n else 0.0 for t,p,n in per_class.values())/len(per_class)
            if 'candidate_hits' in group:
                for k in ('candidate_hits','candidate_size_sum','partial_count','partial_hits'):
                    row[k]=int(group[k].sum())
                row['candidate_hit_rate']=row['candidate_hits']/total
                row['mean_candidate_size']=row['candidate_size_sum']/total
                row['partial_reduction_rate']=row['partial_count']/total
                row['partial_hit_rate']=row['partial_hits']/row['partial_count'] if row['partial_count'] else None
            for k in ('active_rule_count','rules_A1','rules_A2','rules_B1','rules_B2'):
                if k in group:row[k]=float(group[k].mean())
            for k in ['fit_seconds','train_seconds','predict_seconds','preprocessing_fit_seconds','preprocessing_test_seconds',
                      'defer_no_evidence','defer_tie']:
                row[k]=float(group[k].sum())
            if 'illicit_tp' in group:
                tp,fp,fn = (int(group[k].sum()) for k in ('illicit_tp','illicit_fp','illicit_fn'))
                row.update(illicit_tp=tp,illicit_fp=fp,illicit_fn=fn,
                    illicit_precision=tp/(tp+fp) if tp+fp else 0.0,
                    illicit_recall=tp/(tp+fn) if tp+fn else None,
                    illicit_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
                # 此入口每种子只有一个固定时间留出，不拼接不同模型的排序分数。
                row['illicit_average_precision']=float(group.illicit_average_precision.mean())
            if group.rule_count.notna().any():
                row['rule_count']=float(group.rule_count.mean())
        per_seed.append(row)
    seed_frame=pd.DataFrame(per_seed)
    summaries=[]
    metrics=['coverage','classified_accuracy','overall_correct_rate','macro_f1',
             'fit_seconds','train_seconds','predict_seconds','preprocessing_test_seconds','rule_count',
             'defer_no_evidence','defer_tie']
    metrics += ['macro_precision','macro_recall','candidate_hit_rate','mean_candidate_size',
                'partial_reduction_rate','partial_hit_rate','active_rule_count',
                'rules_A1','rules_A2','rules_B1','rules_B2','preprocessing_fit_seconds']
    metrics += ['illicit_precision','illicit_recall','illicit_f1','illicit_average_precision']
    for key,g in seed_frame.groupby(keys[:-1],dropna=False,sort=False):
        row=dict(zip(keys[:-1],key));row.update(expected_seeds=len(g),completed_seeds=int(g.complete.sum()))
        row['complete']=bool(g.complete.all())
        for metric in metrics:
            if metric not in g:
                row[metric+'_defined_seeds']=0
                row[metric+'_mean']=row[metric+'_std']=None
                continue
            values=pd.to_numeric(g[metric],errors='coerce')
            row[metric+'_defined_seeds']=int(values.notna().sum())
            row[metric+'_mean']=values.mean() if row['complete'] else None
            row[metric+'_std']=values.std(ddof=1) if row['complete'] else None
        summaries.append(row)
    # 逐规则统计可复核支持度与测试正确率；不保存冗余的逐样本预测文件。
    tmp=output.with_name(output.stem+'.tmp.xlsx')
    with pd.ExcelWriter(tmp,engine='openpyxl') as writer:
        frame.drop(columns=['per_class'],errors='ignore').to_excel(writer,sheet_name='folds',index=False)
        seed_frame.to_excel(writer,sheet_name='per_seed',index=False)
        pd.DataFrame(summaries).to_excel(writer,sheet_name='summary',index=False)
        pd.DataFrame(states).to_excel(writer,sheet_name='tasks',index=False)
        write_rule_sheets(writer, rules)
    os.replace(tmp,output)
    return all(s['status']=='ok' for s in states)


def run_classification_tasks(tasks, settings, output):
    """公共调度保存原始检查点，本函数负责输出本次任务矩阵的统一报表。"""
    import json
    from pathlib import Path
    from ..common.runner import run_tasks
    from ..common.checkpoints import completed,save
    root=Path(output['root']);root.mkdir(parents=True,exist_ok=True)
    directory=root/output['checkpoint_dir'];template=output['template']
    save(directory/output['index'],{'tasks':tasks})
    for task in tasks:
        path=directory/template.format(task_id=task['id'])
        if settings.get('resume',True) and completed(path,task):
            print(f"[复用] {task['name']} | {task.get('protocol','stratified')} | {task['model_name']} | train={task.get('train_budget',len(task['split']['train_indices']))} | fraction={task.get('fraction','n/a')} | bins={task['max_bins']} | seed={task['split']['seed']} fold={task['split']['fold']}",flush=True)
        elif path.exists():
            # 保留失败尝试，防止过期文件被当作重试后的成功记录。
            path.rename(path.with_name(path.stem+'.previous-'+str(time.time_ns())+'.json'))
    try:
        run_tasks(tasks,classification_worker,directory,settings,template)
    finally:
        passed=summarize_classification(tasks,directory,root/output['results'],template)
    if not passed:raise RuntimeError('Some tasks are incomplete/failed; inspect workbook and checkpoints')
    print(f"结果：{root/output['results']}",flush=True)
