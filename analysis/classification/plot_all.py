"""一条命令生成 UCI/BABD 配图；论文图配置集中于 main。"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from result_data import select_results, curve_data, candidate_data, ablation_data
from figures import (plot_performance_curves, plot_candidate_distribution,
                     plot_ablation_heatmap, plot_computational_cost, plot_training_profile)


def generate(job, settings):
    frame = select_results(job['file'], job['filters'], settings['seeds'])
    common = {k:job[k] for k in ('title','xlabel','ylabel')}
    if job['kind'] in ('performance','cost'):
        common['title'] += f"\nMean ± SD across {len(settings['seeds'])} seeds"
    common['figsize'] = settings['heatmap_size'] if job['kind']=='ablation' else settings['figure_size']
    if job['kind']=='training_profile':
        pieces=[]
        for panel in job['panels']:
            for metric in panel['metrics']:
                piece=curve_data(frame,'train_budget',metric,'voting')
                pieces.append(piece.assign(metric=metric))
        data=pd.concat(pieces,ignore_index=True)
        fig=plot_training_profile(data,panels=job['panels'],title=job['title'],
            xlabel=job['xlabel'],figsize=settings['profile_size'],
            colors=settings['colors'],markers=settings['markers'])
    elif job['kind'] in ('performance','cost'):
        data = curve_data(frame, job['x'], job['metric'], job['series'])
        common.update(x=job['x'], series=job['series'], colors=settings['colors'],
                      markers=settings['markers'], uncertainty=settings['uncertainty'])
        if job['kind']=='performance':
            fig=plot_performance_curves(data, **common, ylim=job['ylim'])
        else:
            fig=plot_computational_cost(data, **common, log_y=settings['log_time'])
    elif job['kind']=='candidate':
        data=candidate_data(frame)
        fig=plot_candidate_distribution(data, **common, colors=settings['candidate_colors'])
    elif job['kind']=='ablation':
        data=ablation_data(frame, job['metric'], settings['reference_rules'])
        fig=plot_ablation_heatmap(data, **common, cmap=settings['heatmap_cmap'],
                                 colorbar_label='Difference from all rules')
    else:
        raise ValueError(f"Unknown figure kind: {job['kind']}")
    # 保存实际绘图数据，便于不重新训练就复现每幅图。
    root=settings['output']; root.mkdir(parents=True,exist_ok=True)
    outputs=[]
    try:
        for extension in settings['formats']:
            target=root/(job['name']+'.'+extension)
            fig.savefig(target,dpi=settings['dpi'])
            outputs.append(target.name)
        table=root/(job['name']+'.csv')
        data.to_csv(table,index=job['kind']=='ablation');outputs.append(table.name)
    finally:
        plt.close(fig)
    return dict(outputs=outputs, selected_seed_rows=len(frame), seeds=sorted(frame.seed.unique().tolist()))


def main(argv=None):
    # 所有可调配置放在这里；函数接口也支持其它数据文件、坐标轴和标题。
    root=Path(__file__).resolve().parents[2]
    settings=dict(output=root/'results/figures/classification', dpi=300, formats=['pdf','png'],
        profile_size=(12.5,7.5), figure_size=(6.4,4.4), heatmap_size=(9,4.6), font_size=10, seeds=[11,37,53],
        colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000'],
        candidate_colors=['#0072B2','#D55E00'], markers=['o','s','^','D','v','P','X'],
        heatmap_cmap='BrBG', reference_rules='all', uncertainty='sd', log_time=False)
    files=dict(uci=root/'results/classification/uci_three_studies.xlsx',
               babd=root/'results/classification/babd_balanced_results.xlsx')
    datasets=['iris','wine','wdbc']  # 新数据集结果完成后，可在此添加 balloons/lenses。
    uci_bins=20
    babd_bins=20
    metrics=[('overall_correct_rate','Overall correct classification rate',(0,1)),
             ('candidate_hit_rate','Candidate-set hit rate',(0,1)),
             ('mean_candidate_size','Mean candidate-set size',(0,None))]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uci',type=Path,default=files['uci'])
    parser.add_argument('--babd',type=Path,default=files['babd'])
    parser.add_argument('--output',type=Path,default=settings['output'])
    parser.add_argument('--dpi',type=int,default=settings['dpi'])
    args=parser.parse_args(argv)
    if args.dpi<=0:parser.error('dpi must be positive')
    files.update(uci=args.uci,babd=args.babd);settings.update(output=args.output,dpi=args.dpi)
    plt.rcParams.update({'font.size':settings['font_size'],'pdf.fonttype':42,'ps.fonttype':42})
    jobs=[]
    for dataset in datasets:
        for metric,label,limits in metrics:
            jobs.append(dict(name=f'uci_{dataset}_{metric}',kind='performance',file=files['uci'],
                filters=dict(study='sensitivity',dataset=dataset,model='rc-oeo',rule_group='all'),
                x='max_bins',metric=metric,series='voting',title=f'{dataset.upper()}: binning and voting',
                xlabel='Maximum number of bins',ylabel=label,ylim=limits))
        jobs.append(dict(name=f'uci_{dataset}_candidates',kind='candidate',file=files['uci'],
            filters=dict(study='comparison',dataset=dataset,model='rc-oeo',max_bins=uci_bins,
                         voting='uniform',rule_group='all'),
            title=f'{dataset.upper()}: candidate outcomes (bins={uci_bins})',
            xlabel='Candidate-set group',ylabel='Proportion of test predictions'))
    for metric,label,_ in metrics:
        jobs.append(dict(name=f'uci_ablation_{metric}',kind='ablation',file=files['uci'],
            filters=dict(study='ablation',dataset=datasets,model='rc-oeo',max_bins=uci_bins,voting='uniform'),
            metric=metric,title=f'Rule ablation: {label.lower()}',xlabel='Rule combination',ylabel='Dataset'))
    # BABD 不同协议或预算不能混合；此处绘制已完成的 balanced 数据。
    for metric,label in [('fit_seconds','Model fit time (s)'),('predict_seconds','Prediction time (s)')]:
        jobs.append(dict(name=f'babd_{metric}',kind='cost',file=files['babd'],
            filters=dict(dataset='babd_balanced',protocol='balanced_holdout',max_bins=babd_bins,
                         model='rc-oeo',voting='uniform'),
            x='train_budget',series='voting',metric=metric,title=f'BABD: RC-OEO (bins={babd_bins})',
            xlabel='Training objects',ylabel=label))
    # 核心方法评估图：固定协议、分箱上限和投票，展示实际变化而非单调性假设。
    profile_panels=[
        dict(title='Extracted rules',ylabel='Number of rules',ylim=(0,None),
             metrics={'rule_count':'All rules'}),
        dict(title='Accuracy of unique predictions',ylabel='Classified accuracy',ylim=(0,1),
             metrics={'classified_accuracy':'Classified accuracy'}),
        dict(title='Coverage and correct predictions',ylabel='Proportion of test objects',ylim=(0,1),
             metrics={'coverage':'Unique-class coverage','overall_correct_rate':'Overall correct rate'}),
        dict(title='True class retained',ylabel='Candidate-set hit rate',ylim=(0,1),
             metrics={'candidate_hit_rate':'Candidate hit rate'}),
        dict(title='Remaining decision scope',ylabel='Mean candidate-set size',ylim=(0,None),
             metrics={'mean_candidate_size':'Mean set size'}),
        dict(title='Partial decisions',ylabel='Partial reduction rate',ylim=(0,1),
             metrics={'partial_reduction_rate':'Partial reduction rate'})]
    jobs.append(dict(name='babd_training_profile',kind='training_profile',file=files['babd'],
        filters=dict(dataset='babd_balanced',protocol='balanced_holdout',model='rc-oeo',
                     max_bins=babd_bins,voting='uniform'),
        panels=profile_panels,xlabel='Training objects',ylabel='',
        title=f'RC-OEO: training size, rules and decisions (BABD balanced; bins={babd_bins})'
              +f"\nMean ± SD across {len(settings['seeds'])} seeds; encoding fitted at each training size"))
    settings['output'].mkdir(parents=True,exist_ok=True)
    manifest=dict(settings=settings, versions=dict(python=platform.python_version(),
        pandas=pd.__version__,matplotlib=matplotlib.__version__),figures=[],
        notes=['Curves: equal-seed mean and sample SD; SD is not a confidence interval.',
               'Candidates: exact grouped counts, not an exact size histogram.',
               'Fit times shared across voting variants are not independent fits.',
               'Per-seed time is the sum over its folds, as recorded in the source workbook.'])
    manifest['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('plot_all.py','figures.py','result_data.py')}
    hashes={}
    for job in jobs:
        record=dict(job)
        try:
            if job['file'] not in hashes:
                hashes[job['file']]=hashlib.sha256(job['file'].read_bytes()).hexdigest()
            record.update(source_sha256=hashes[job['file']],**generate(job,settings),status='ok')
            print('[生成]',job['name'])
        except (ValueError,KeyError,FileNotFoundError) as error:
            record.update(status='skipped',reason=str(error));print('[跳过]',job['name'],error)
        manifest['figures'].append(record)
    (settings['output']/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    return 1 if any(r['status']!='ok' for r in manifest['figures']) else 0


if __name__=='__main__':
    raise SystemExit(main())
