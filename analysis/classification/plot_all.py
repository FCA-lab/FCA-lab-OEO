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
from result_data import (babd_timing_data, babd_behavior_data, select_results,
                         comparison_grid_data, ablation_grid_data,
                         voting_comparison_data, candidate_size_data)
from figures import (plot_babd_timing, plot_babd_behavior, plot_comparison_grid,
                     plot_ablation_combined, plot_voting_comparison, plot_candidate_sizes)


def generate(job, settings):
    frame = select_results(job['file'], job['filters'], settings['seeds'])
    if job['kind'] in ('babd_behavior','babd_timing'):
        frame['experiment']='balanced'
        other=select_results(job['imbalanced_file'],job['filters'],settings['seeds'])
        other['experiment']='imbalanced'
        frame=pd.concat([frame,other],ignore_index=True)
    if job['kind']=='babd_behavior':
        data=babd_behavior_data(frame)
        fig=plot_babd_behavior(data)
    elif job['kind']=='babd_timing':
        data=babd_timing_data(frame)
        fig=plot_babd_timing(data)
    elif job['kind']=='comparison_grid':
        data=comparison_grid_data(frame, job['datasets'], job['models'], job['bins'], job['metrics'])
        fig=plot_comparison_grid(data,datasets=job['datasets'],models=job['models'],
            metrics=job['metrics'],figsize=(7.2,4.8),colors=settings['colors'],
            markers=settings['markers'],ylim=job['ylim'])
    elif job['kind']=='candidate_sizes':
        data=candidate_size_data(frame,job['datasets'],settings['seeds'])
        fig=plot_candidate_sizes(data,datasets=job['datasets'])
    elif job['kind']=='voting_comparison':
        data=voting_comparison_data(frame,job['datasets'],job['bins'],job['metrics'],settings['seeds'])
        fig=plot_voting_comparison(data,datasets=job['datasets'])
    elif job['kind']=='ablation_combined':
        data=ablation_grid_data(frame, job['datasets'], job['rule_groups'],
            job['metrics'], settings['seeds'], job['numerical_bins'])
        fig=plot_ablation_combined(data, datasets=job['datasets'], rule_groups=job['rule_groups'],
            metrics=job['metrics'])
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
        data.to_csv(table,index=False);outputs.append(table.name)
    finally:
        plt.close(fig)
    return dict(outputs=outputs, selected_seed_rows=len(frame), seeds=sorted(frame.seed.unique().tolist()))


def main(argv=None):
    # 所有可调配置放在这里；函数接口也支持其它数据文件、坐标轴和标题。
    root=Path(__file__).resolve().parents[2]
    settings=dict(output=root/'results/figures/classification', dpi=300, formats=['pdf','png'],
        font_size=10, seeds=[11,37,53],
        colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000'],
        markers=['o','s','^','D','v','P','X'])
    files=dict(uci=root/'results/classification/uci_three_studies.xlsx',
               babd=root/'results/classification/babd_balanced_results.xlsx')
    datasets=['iris','wine','wdbc']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uci',type=Path,default=files['uci'])
    parser.add_argument('--babd',type=Path,default=files['babd'])
    parser.add_argument('--babd-imbalanced',type=Path,default=root/'results/classification/babd_imbalanced_results.xlsx')
    parser.add_argument('--output',type=Path,default=settings['output'])
    parser.add_argument('--dpi',type=int,default=settings['dpi'])
    parser.add_argument('--only',nargs='+',help='Generate only these named figures; no experiments are run')
    args=parser.parse_args(argv)
    if args.dpi<=0:parser.error('dpi must be positive')
    files.update(uci=args.uci,babd=args.babd);settings.update(output=args.output,dpi=args.dpi)
    plt.rcParams.update({'font.size':settings['font_size'],'pdf.fonttype':42,'ps.fonttype':42})
    jobs=[dict(name='uci_model_comparison_grid',kind='comparison_grid',file=files['uci'],
        filters=dict(study='comparison',dataset=datasets,
            model=['decision_tree','random_forest_context','xgboost_context','rc-oeo'],
            voting=['baseline','uniform']),
        datasets=dict(iris='Iris',wine='Wine',wdbc='WDBC'),
        models={'decision_tree':'DT','random_forest_context':'RF',
                'xgboost_context':'XGBoost','rc-oeo':r'RC-OEO (uniform voting, $RS_{DK}$)'},
        bins=list(range(20,101,10)),
        metrics={'classified_accuracy':'Classified accuracy','macro_precision':'Macro-Precision'},
        ylim=(50,103),title='UCI model comparison',xlabel='Maximum number of bins',ylabel='')]

    jobs.append(dict(name='uci_rule_ablation_combined',kind='ablation_combined',file=files['uci'],
        filters=dict(study='ablation',model='rc-oeo',voting='uniform',
            dataset=['iris','wine','lenses','wdbc','balloons']),
        datasets=dict(iris='Iris',wine='Wine',lenses='Lenses',wdbc='WDBC',balloons='Balloons'),
        rule_groups={'all':'All', 'support_only':r'$A_1,A_2$',
            'exclusion_only':r'$B_1,B_2$', 'without_A1':r'All $\setminus\{A_1\}$',
            'without_A2':r'All $\setminus\{A_2\}$', 'without_B1':r'All $\setminus\{B_1\}$',
            'without_B2':r'All $\setminus\{B_2\}$'},
        metrics={'classified_accuracy':'Classified accuracy','coverage':'Coverage'},
        numerical_bins=20,title='UCI rule ablation',xlabel='Rule combination',ylabel='Dataset'))

    jobs.append(dict(name='uci_candidate_set_sizes',kind='candidate_sizes',file=files['uci'],
        filters=dict(study='comparison',dataset=['iris','wine','lenses'],model='rc-oeo',
                     rule_group='all',voting='uniform',max_bins=[20,float('nan')]),
        datasets=dict(iris='Iris',wine='Wine',lenses='Lenses'),title='',xlabel='',ylabel=''))
    jobs.append(dict(name='uci_voting_model_comparison',kind='voting_comparison',file=files['uci'],
        filters=dict(study=['comparison','sensitivity'],dataset=['iris','wine','wdbc'],
                     model=['decision_tree','random_forest_context','xgboost_context','rc-oeo']),
        datasets=dict(iris='Iris',wine='Wine',wdbc='WDBC'),bins=list(range(20,101,10)),
        metrics=['classified_accuracy','coverage'],title='',xlabel='',ylabel=''))
    jobs.append(dict(name='babd_behavior_grid',kind='babd_behavior',file=files['babd'],
        imbalanced_file=args.babd_imbalanced,
        filters=dict(model='rc-oeo',max_bins=20,voting='uniform'),
        title='',xlabel='',ylabel=''))
    jobs.append(dict(name='babd_runtime_comparison',kind='babd_timing',file=files['babd'],
        imbalanced_file=args.babd_imbalanced,
        filters=dict(model=['rc-oeo','random_forest','xgboost'],max_bins=20),
        title='',xlabel='',ylabel=''))
    if args.only:
        unknown=set(args.only)-{j['name'] for j in jobs}
        if unknown: parser.error(f'Unknown figure names: {sorted(unknown)}')
        jobs=[j for j in jobs if j['name'] in args.only]
    settings['output'].mkdir(parents=True,exist_ok=True)
    manifest=dict(settings=settings, versions=dict(python=platform.python_version(),
        pandas=pd.__version__,matplotlib=matplotlib.__version__),figures=[],
        notes=['Curves: equal-seed mean and sample SD; SD is not a confidence interval.',
               'Candidate size panels use recorded counts and true-class inclusion.',
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
    (settings['output']/('manifest_selected.json' if args.only else 'manifest.json')).write_text(json.dumps(manifest,indent=2,default=str))
    return 1 if any(r['status']!='ok' for r in manifest['figures']) else 0


if __name__=='__main__':
    raise SystemExit(main())
