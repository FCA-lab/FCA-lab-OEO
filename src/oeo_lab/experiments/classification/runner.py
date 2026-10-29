"""Schedule classification tasks and aggregate recorded results by seed."""
import os
import time
import pandas as pd

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
    # 参数组合只引用共享划分，避免重复保存十万级测试样本编号。
    from ..common.task_index import save_task_index
    save_task_index(directory/output['index'], tasks)
    # Recover legacy filenames before dispatch; successful scientific results are unchanged.
    if settings.get("resume", True) and any(t.get("name", "").startswith("babd_") for t in tasks):
        from ..common.checkpoints import recover_compatible_checkpoints
        recover_compatible_checkpoints(tasks, directory, template)
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
