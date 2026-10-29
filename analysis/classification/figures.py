"""论文绘图函数集中于本模块。只绘图，不读文件、不筛选实验、不计算实验指标。"""
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison_grid(data, *, datasets, models, metrics, figsize,
                         colors, markers, ylim=(50, 103)):
    """数据集为列、指标为行；共享模型图例，均值曲线及种子间样本 SD。"""
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    styles = ['-', '--', '-.', ':']
    for row, (metric, ylabel) in enumerate(metrics.items()):
        for col, (dataset, label) in enumerate(datasets.items()):
            ax = axes[row, col]
            for i, (model, name) in enumerate(models.items()):
                group = data[data.dataset.eq(dataset) & data.model.eq(model)
                             & data.metric.eq(metric)].sort_values('max_bins')
                if group.empty or group['std'].isna().any():
                    raise ValueError(f'Missing mean/SD: {dataset}/{model}/{metric}')
                x = group.max_bins.to_numpy()
                mean, sd = group['mean'].to_numpy()*100, group['std'].to_numpy()*100
                if (mean-sd < ylim[0]).any() or (mean+sd > ylim[1]).any():
                    raise ValueError('Axis limits would clip an SD band')
                color = colors[i % len(colors)]
                ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.10, linewidth=0)
                ax.plot(x, mean, label=name, color=color, linestyle=styles[i % len(styles)],
                        marker=markers[i % len(markers)], markersize=3.5, linewidth=1.2,
                        zorder=4 if model=='rc-oeo' else 3)
            ax.set_title(f'({chr(97+row*len(datasets)+col)}) {label}', loc='left', fontsize=9)
            ax.set_xlim(17, 103)
            ax.set_ylim(*ylim)
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.set_xticks(list(range(20,101,10)), minor=True)
            ax.set_yticks([50,60,70,80,90,100])
            ax.tick_params(labelsize=8)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y', alpha=.22, linewidth=.5)
            if col==0: ax.set_ylabel(ylabel+' (%)', fontsize=9)
            if row==len(metrics)-1: ax.set_xlabel('Maximum number of bins', fontsize=8)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52,1),
               ncol=len(models), frameon=False, fontsize=9, handlelength=2.5)
    fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.88,wspace=.16,hspace=.34)
    return fig


def plot_ablation_combined(data, *, datasets, rule_groups, metrics, figsize=(7.2,4.8)):
    """Compact five-row layout at the comparison figure's exact physical size."""
    from matplotlib.ticker import MaxNLocator
    rules=list(rule_groups)
    labels=[r'$\{A_1,A_2,B_1,B_2\}$',r'$\{A_1,A_2\}$',r'$\{B_1,B_2\}$',
            r'$\{A_2,B_1,B_2\}$',r'$\{A_1,B_1,B_2\}$',r'$\{A_1,A_2,B_2\}$',r'$\{A_1,A_2,B_1\}$']
    colors=['#222222','#0072B2','#D55E00','#009E73','#CC79A7','#B8860B','#56B4E9']
    fig,axes=plt.subplots(len(datasets),len(metrics),figsize=figsize,squeeze=False)
    for row,(dataset,name) in enumerate(datasets.items()):
        for col,(metric,title) in enumerate(metrics.items()):
            ax=axes[row,col]
            p=data[data.dataset.eq(dataset)&data.metric.eq(metric)].set_index('rule_group').reindex(rules)
            mean=p.mean_pct.to_numpy();sd=p.sd_pct.to_numpy();valid=np.isfinite(mean)&np.isfinite(sd)
            lo=np.min(mean[valid]-sd[valid]);hi=np.max(mean[valid]+sd[valid]);pad=max((hi-lo)*.12,.5)
            ax.set_xlim((0,100) if dataset=='balloons' else (lo-pad,hi+pad))
            ax.axhspan(-.48,.48,color='#edf0f3',zorder=0)
            for i in range(len(rules)):
                if valid[i]:
                    ax.errorbar(mean[i],i,xerr=sd[i],fmt='D' if i==0 else 'o',color=colors[i],
                                markersize=3.2 if i==0 else 2.7,capsize=1.8,elinewidth=.85,
                                markeredgecolor='white',markeredgewidth=.3,clip_on=False,zorder=3)
            ax.set_ylim(6.6,-.6)
            ax.set_yticks(range(7),labels if col==0 else ['']*7,fontsize=6.5)
            ax.tick_params(axis='y',length=0,pad=5)
            ax.tick_params(axis='x',labelsize=7,length=2,pad=2)
            if dataset=='balloons':ax.set_xticks([0,20,40,60,80,100])
            else:ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            if dataset == 'lenses' and metric == 'classified_accuracy':
                # Keep the full SD interval; only restrict percentage tick labels.
                limits = ax.get_xlim()
                ax.set_xticks([tick for tick in ax.get_xticks() if limits[0] <= tick <= min(100, limits[1])])
                ax.set_xlim(limits)
            ax.grid(axis='x',color='#e4e4e4',linewidth=.5)
            ax.axhline(2.5,color='#dddddd',linewidth=.5)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color('#aaaaaa')
            if row==len(datasets)-1:ax.set_xlabel(title+' (%)',fontsize=9,labelpad=6)
            if col==0:ax.set_ylabel(name,fontsize=8,labelpad=12,fontweight='normal')
    fig.subplots_adjust(left=.21,right=.97,top=.97,bottom=.12,wspace=.20,hspace=.43)
    return fig


def plot_voting_comparison(data, *, datasets, figsize=(7.2,4.8)):
    from matplotlib.lines import Line2D
    series=['DT','RF','XGBoost','RC-OEO: uniform','RC-OEO: nonzero','RC-OEO: support-weighted']
    colors=['#8da9bb','#b3b3b3','#bca992','#0072B2','#0072B2','#0072B2']
    styles=['-','--','-.','-','--',':'];markers=['o','s','^','o','s','D']
    fig,axes=plt.subplots(2,3,figsize=figsize,sharex=True,squeeze=False)
    for col,(dataset,name) in enumerate(datasets.items()):
        for row,metric in enumerate(['classified_accuracy','coverage']):
            ax=axes[row,col]
            for i,label in enumerate(series):
                if row==1 and i<3:continue
                g=data[data.dataset.eq(dataset)&data.metric.eq(metric)&data.series.eq(label)].sort_values('max_bins')
                x=g.max_bins.to_numpy();mean=g['mean'].to_numpy()*100;sd=g['std'].to_numpy()*100
                ax.fill_between(x,mean-sd,mean+sd,color=colors[i],alpha=.055 if i<3 else .07,linewidth=0)
                ax.plot(x,mean,color=colors[i],linestyle=styles[i],marker=markers[i],
                    markerfacecolor='white' if i>=4 else colors[i],markeredgewidth=.8,
                    markersize=3.3,linewidth=1 if i<3 else 1.3,zorder=3+i*.1)
            if row==1:ax.axhline(100,color='#999999',linestyle='--',linewidth=1)
            ax.set_ylim((50,103) if row==0 else (80,102))
            ax.set_yticks([50,60,70,80,90,100] if row==0 else [80,85,90,95,100])
            ax.set_xlim(17,103);ax.set_xticks([20,40,60,80,100])
            ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.2,linewidth=.5)
            ax.spines[['top','right']].set_visible(False)
            ax.set_title(f'({chr(97+row*3+col)}) {name}',loc='left',fontsize=9)
            if col==0:ax.set_ylabel('Classified accuracy (%)' if row==0 else 'Coverage (%)',fontsize=9)
            if row==1:ax.set_xlabel('Maximum number of bins',fontsize=8)
    handles=[Line2D([],[],color=colors[i],linestyle=styles[i],marker=markers[i],markersize=3.3,
             markerfacecolor='white' if i>=4 else colors[i],label=label) for i,label in enumerate(series)]
    # Column-major legend order gives baseline models above the RC-OEO variants.
    order=[0,3,1,4,2,5]
    fig.legend(handles=[handles[i] for i in order],loc='upper center',ncol=3,frameon=False,
               fontsize=8,bbox_to_anchor=(.54,1),columnspacing=1.8,handlelength=2.5)
    axes[1,2].text(.97,.97,'DT / RF / XGBoost: 100%',transform=axes[1,2].transAxes,
                   ha='right',va='top',fontsize=6.5,color='#666666')
    fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.82,wspace=.17,hspace=.35)
    return fig


def plot_candidate_sizes(data, *, datasets, figsize=(7.2,3.5)):
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(1,len(datasets),figsize=figsize,sharey=True,squeeze=False)
    for col,(dataset,label) in enumerate(datasets.items()):
        ax=axes[0,col];part=data[data.dataset.eq(dataset)]
        g=part.groupby('candidate_size')[['retained_pct','excluded_pct','total_pct']].mean().reindex([1,2,3])
        ax.bar([1,2,3],g.retained_pct,width=.58,color='#0072B2',zorder=3)
        ax.bar([1,2,3],g.excluded_pct,bottom=g.retained_pct,width=.58,
               color='#D55E00',edgecolor='white',hatch='///',linewidth=.5,zorder=3)
        for size in (1,2,3):
            ax.text(size,g.loc[size,'total_pct']+2,f"{g.loc[size,'total_pct']:.2f}%",ha='center',va='bottom',fontsize=8)
        ax.set_title(f'({chr(97+col)}) {label}',loc='left',fontsize=9)
        ax.set_xticks([1,2,3]);ax.set_xlabel('Candidate-set size',fontsize=9)
        ax.set_ylim(0,108);ax.set_xlim(.4,3.6);ax.set_yticks([0,20,40,60,80,100])
        ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2,linewidth=.5,zorder=0)
        if col==0:ax.set_ylabel('Test objects (%)',fontsize=9)
    fig.legend(handles=[Patch(facecolor='#0072B2',label='True class included'),
                        Patch(facecolor='#D55E00',hatch='///',edgecolor='white',label='True class not included')],
               loc='upper center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.55,1))
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.80,wspace=.20)
    return fig


def plot_babd_behavior(data, figsize=(7.2, 4.8)):
    """Training-size behavior; points and error bars are seed means and sample SD."""
    from matplotlib.ticker import FuncFormatter
    with plt.rc_context({'font.size':7, 'axes.labelsize':7, 'xtick.labelsize':6.5,
                         'ytick.labelsize':6.5}):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.subplots_adjust(left=.105,right=.98,bottom=.13,top=.88,wspace=.42,hspace=.48)
        blue, orange, green = '#0072B2', '#D55E00', '#009E73'
        for row, (key, label) in enumerate([('balanced','Balanced (7 classes)'),
                                           ('imbalanced','Imbalanced (11 classes)')]):
            d=data[data.experiment.eq(key)].sort_values('train_budget')
            x=d.train_budget.to_numpy()
            for ax in axes[row]:
                ax.spines[['top','right']].set_visible(False)
                ax.grid(color='#dddddd', linewidth=.45)
                ax.set_axisbelow(True)
            for col in [0,1]:
                ax=axes[row,col]
                ax.set_xscale('log')
                ax.set_xticks(x)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v/1000:g}k' if v>=1000 else f'{v:g}'))
                ax.minorticks_off()
                ax.tick_params(axis='x',rotation=45)
                ax.set_xlabel('Training samples (log scale)')
            ax=axes[row,0]
            ax.errorbar(x,d.rule_count_mean,yerr=d.rule_count_std,color=blue,
                        marker='o',ms=3,lw=1.1,capsize=2)
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Extracted rules')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v/1000:g}k'))
            axes[row,0].text(-.36,.5,label,transform=axes[row,0].transAxes,
                            rotation=90,va='center',ha='center',fontsize=8)
            ax=axes[row,1]
            for metric,color,marker,ls,label in [
                ('classified_accuracy',blue,'o','-','Classified accuracy'),
                ('coverage',orange,'s','--','Coverage')]:
                ax.errorbar(x,100*d[metric+'_mean'],yerr=100*d[metric+'_std'],
                            color=color,marker=marker,ls=ls,ms=3,lw=1.1,capsize=2,label=label)
            ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Percentage (%)')
            if row==0:ax.legend(loc='center',bbox_to_anchor=(.55,.45),fontsize=6,frameon=False,handlelength=1.5)
            ax=axes[row,2]
            ax.errorbar(d.mean_candidate_size_mean,100*d.candidate_hit_rate_mean,
                        xerr=d.mean_candidate_size_std,yerr=100*d.candidate_hit_rate_std,
                        color=green,marker='o',ms=3,lw=1.1,capsize=2)
            ax.set_xlim(.8,5.4);ax.set_xticks([1,2,3,4,5])
            ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100])
            ax.set_xlabel('Mean candidate-set size');ax.set_ylabel('True-class inclusion (%)')
            for i,offset in [(0,(5,-13)),(-1,(-4,-15))]:
                item=d.iloc[i]
                ax.annotate(f'n={int(item.train_budget):,}',
                    (item.mean_candidate_size_mean,100*item.candidate_hit_rate_mean),
                    xytext=offset,textcoords='offset points',fontsize=6.5,
                    ha='left' if i==0 else 'right')
        for ax,title in zip(axes[0],['Rule acquisition','Unique predictions','Candidate sets']):
            ax.set_title(title,fontsize=8,pad=10)
        return fig


def plot_babd_timing(data, figsize=(7.2,3.3)):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    styles=[('rc-oeo','RC-OEO','#0072B2','o'),
            ('random_forest','RF','#D55E00','s'),('xgboost','XGBoost','#009E73','^')]
    with plt.rc_context({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7}):
        fig,axes=plt.subplots(1,2,figsize=figsize)
        fig.subplots_adjust(left=.095,right=.98,bottom=.21,top=.79,wspace=.36)
        for ax,metric,ylabel in zip(axes,['fit_per_fold','predict_per_1000'],
                ['Rule extraction / model fitting (s)', 'Prediction per 1,000 objects (s)']):
            for model,label,color,marker in styles:
                for experiment,ls in [('balanced','-'),('imbalanced','--')]:
                    d=data[(data.model==model)&(data.experiment==experiment)].sort_values('train_budget')
                    ax.errorbar(d.train_budget,d[metric+'_mean'],yerr=d[metric+'_std'],
                        color=color,marker=marker,ls=ls,ms=3.5,lw=1.1,capsize=2,
                        markerfacecolor=color if experiment=='balanced' else 'white')
            ax.set_xscale('log');ax.set_yscale('log')
            ax.set_xticks([500,1000,2000,5000,10000,30000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:g}k' if v>=1000 else f'{v:g}'))
            ax.set_xlabel('Training samples (log scale)')
            ax.set_ylabel(ylabel+'\n(log scale)')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(which='major',color='#dddddd',lw=.5);ax.set_axisbelow(True)
        handles=[Line2D([],[],color=c,marker=m,lw=1.1,label=l) for _,l,c,m in styles]
        handles += [Line2D([],[],color='#444444',ls=ls,label=l) for l,ls in
                    [('Balanced (7 classes)','-'),('Imbalanced (11 classes)','--')]]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),
                   ncol=3,frameon=False,fontsize=7,columnspacing=1.5)
        return fig
