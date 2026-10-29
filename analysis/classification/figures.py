"""论文绘图函数集中于本模块。只绘图，不读文件、不筛选实验、不计算实验指标。"""
import numpy as np
import matplotlib.pyplot as plt


def _axes(title, xlabel, ylabel, figsize):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[['top','right']].set_visible(False)
    return fig, ax


def plot_performance_curves(data, *, x, series, title, xlabel, ylabel, figsize,
                            colors, markers, ylim=None, uncertainty='sd'):
    """data 包含系列、x、mean/std/n；误差线为种子间 SD，不是置信区间。"""
    fig, ax = _axes(title, xlabel, ylabel, figsize)
    for i, (name, group) in enumerate(data.groupby(series, sort=False)):
        group = group.sort_values(x)
        error = group['std'].fillna(0) if uncertainty == 'sd' else None
        ax.errorbar(group[x], group['mean'], yerr=error, label=str(name),
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    capsize=3, linewidth=1.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    return fig


def plot_candidate_distribution(data, *, title, xlabel, ylabel, figsize, colors,
                                hit_label='True class retained', miss_label='True class absent'):
    """category 可为精确集合大小或明确标注的分组；hit/miss 为全测试集比例。"""
    fig, ax = _axes(title, xlabel, ylabel, figsize)
    ax.bar(data.category.astype(str), data.hit, color=colors[0], label=hit_label)
    ax.bar(data.category.astype(str), data.miss, bottom=data.hit, color=colors[1],
           hatch='//', label=miss_label)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return fig


def plot_ablation_heatmap(data, *, title, xlabel, ylabel, figsize, cmap, colorbar_label):
    """data 为配对差值矩阵；零为中心，不将缺失结果填成零。"""
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    values = data.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 1e-9)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-bound, vmax=bound, aspect='auto')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(data.index)), data.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v=values[i,j]
            ax.text(j,i, f'{v:+.3f}' if np.isfinite(v) else 'N/A', ha='center', va='center',
                    color='white' if abs(v)>bound*.65 else 'black')
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return fig


def plot_computational_cost(data, *, x, series, title, xlabel, ylabel, figsize,
                            colors, markers, log_y=False, uncertainty='sd'):
    """时间口径由入口明确指定；不把任务总耗时视为模型训练时间。"""
    fig = plot_performance_curves(data, x=x, series=series, title=title,
        xlabel=xlabel, ylabel=ylabel, figsize=figsize, colors=colors, markers=markers,
        uncertainty=uncertainty)
    ax=fig.axes[0]
    if log_y:
        if (data['mean'] <= 0).any():
            raise ValueError('Log time axis requires positive values')
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)
    return fig


def plot_training_profile(data, *, panels, title, xlabel, figsize, colors, markers):
    """训练规模核心组合图；各指标独立纵轴，不使用双轴或预设趋势。"""
    fig, axes=plt.subplots(2,3,figsize=figsize,layout='constrained')
    fig.suptitle(title)
    for index,(ax,panel) in enumerate(zip(axes.flat,panels)):
        for i,(metric,label) in enumerate(panel['metrics'].items()):
            group=data[data.metric.eq(metric)].sort_values('train_budget')
            if group.empty:
                raise ValueError(f'Missing training-profile metric: {metric}')
            ax.errorbar(group.train_budget,group['mean'],yerr=group['std'].fillna(0),
                        label=label,color=colors[i%len(colors)],marker=markers[i%len(markers)],
                        capsize=3,linewidth=1.4)
        ax.set(title=f"({chr(97+index)}) {panel['title']}",xlabel=xlabel,ylabel=panel['ylabel'])
        ax.set_ylim(*panel['ylim'])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        if len(panel['metrics'])>1:ax.legend(frameon=False,fontsize=8)
    return fig


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


def plot_ablation_grid(data, *, datasets, rule_groups, metrics, figsize=(7.2, 5.6), cmap='BrBG'):
    """Cell text: mean percent; shared colour scale: paired change in percentage points."""
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, layout='constrained', squeeze=False)
    finite = data.delta_pp.dropna().abs()
    bound = max(5.0, float(np.ceil(finite.max()/5)*5)) if len(finite) else 5.0
    norm = Normalize(vmin=-bound, vmax=bound)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad('#dddddd')
    for panel, (metric, title) in enumerate(metrics.items()):
        ax = axes[panel, 0]
        part = data[data.metric.eq(metric)]
        means = part.pivot(index='dataset', columns='rule_group', values='mean_pct').reindex(index=datasets, columns=rule_groups).to_numpy()
        delta = part.pivot(index='dataset', columns='rule_group', values='delta_pp').reindex(index=datasets, columns=rule_groups).to_numpy()
        im = ax.imshow(np.ma.masked_invalid(delta), cmap=palette, norm=norm, aspect='auto')
        ax.set_title(f'({chr(97+panel)}) Mean {title.lower()} (%)', loc='left', fontsize=10, pad=8)
        ax.set_xticks(range(len(rule_groups)), list(rule_groups.values()), fontsize=8)
        ax.set_yticks(range(len(datasets)), list(datasets.values()), fontsize=9)
        ax.tick_params(length=0, pad=6)
        ax.set_xticks(np.arange(-.5, len(rule_groups), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(datasets), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)
        ax.axvline(.5, color='#777777', linewidth=1)
        ax.axhline(2.5, color='#777777', linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(datasets)):
            for j in range(len(rule_groups)):
                value = means[i,j]
                rgba = palette(norm(delta[i,j])) if np.isfinite(delta[i,j]) else (.87,.87,.87,1)
                rgb = np.asarray(rgba[:3])
                linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
                luminance = linear @ np.array([.2126,.7152,.0722])
                color = 'black' if luminance > .179 else 'white'
                ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else '—',
                    ha='center', va='center', fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=list(axes[:,0]), fraction=.035, pad=.025, shrink=.9)
    cbar.set_label('Change from all rules (percentage points)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig


def plot_ablation_dots(data, *, datasets, rule_groups, metrics, figsize=(9, 4.8)):
    """Paired seed differences from all rules; horizontal bars show sample SD, not CI."""
    from matplotlib.lines import Line2D
    variants = [rule for rule in rule_groups if rule != 'all']
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=figsize,
                             sharey=True, squeeze=False)
    colors = ['#0072B2', '#D55E00']
    y = np.arange(len(variants))
    for row, (metric, metric_label) in enumerate(metrics.items()):
        values = data[data.metric.eq(metric) & data.rule_group.isin(variants)]
        lo = (values.delta_pp-values.delta_sd_pp).min()
        hi = (values.delta_pp+values.delta_sd_pp).max()
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -5., 5.
        lower = min(-5., np.floor(lo/5)*5-1)
        upper = max(5., np.ceil(hi/5)*5+1)
        for col, (dataset, label) in enumerate(datasets.items()):
            ax = axes[row, col]
            part = data[data.metric.eq(metric) & data.dataset.eq(dataset)].set_index('rule_group')
            group = part.reindex(variants)
            mean = group.delta_pp.to_numpy(dtype=float)
            sd = group.delta_sd_pp.to_numpy(dtype=float)
            valid = np.isfinite(mean) & np.isfinite(sd)
            ax.axvline(0, color='#777777', linewidth=.8, zorder=1)
            ax.hlines(y[valid], 0, mean[valid], color=colors[row % 2], linewidth=1.5, alpha=.5)
            ax.errorbar(mean[valid], y[valid], xerr=sd[valid], fmt='o', color=colors[row % 2],
                        markersize=4, elinewidth=.8, capsize=2, zorder=3)
            for pos in y[~valid]:
                ax.text(.5, pos, 'N/A', transform=ax.get_yaxis_transform(), ha='center', fontsize=7)
            baseline = part.loc['all', 'mean_pct']
            reference = f'{baseline:.2f}%' if np.isfinite(baseline) else 'N/A'
            ax.set_title(f'{label}\nAll: {reference}', fontsize=8, pad=7)
            ax.set_xlim(lower, upper)
            ax.set_ylim(len(variants)-.5, -.5)
            ax.set_yticks(y, [rule_groups[k] for k in variants], fontsize=8)
            ax.tick_params(axis='both', labelsize=8, length=2)
            ax.spines[['top','right','left']].set_visible(False)
            ax.grid(axis='y', color='#eeeeee', linewidth=.6)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=9, labelpad=9)
    handles = [Line2D([], [], color=colors[0], marker='o', linestyle='none', markersize=4,
                      label='Mean paired change; error bars: sample SD across 3 seeds')]
    fig.legend(handles=handles, loc='upper center', frameon=False, fontsize=8,
               bbox_to_anchor=(.56, 1.0))
    fig.supxlabel('Change from all rules (percentage points)', fontsize=9, y=.02)
    fig.subplots_adjust(left=.16, right=.985, top=.85, bottom=.12, wspace=.28, hspace=.65)
    return fig


def plot_ablation_tradeoff(data, *, datasets, rule_groups, figsize=(9, 5.5)):
    """Absolute mean accuracy versus coverage; coincident means merged, not predictions."""
    from matplotlib.lines import Line2D
    rules = list(rule_groups)
    colors = ['#222222','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    markers = ['*','o','s','^','D','v','P']
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for index, (dataset, label) in enumerate(datasets.items()):
        ax = axes.flat[index]
        part = data[data.dataset.eq(dataset)].pivot(index='rule_group', columns='metric', values='mean_pct')
        groups = []
        for k, rule in enumerate(rules):
            xy = part.loc[rule, ['coverage','classified_accuracy']].to_numpy(dtype=float)
            if not np.isfinite(xy).all():
                continue
            match = next((g for g in groups if np.allclose(g['xy'],xy,rtol=0,atol=1e-10)),None)
            if match is None:
                groups.append(dict(xy=xy, ids=[k]))
            else:
                match['ids'].append(k)
        base = part.loc['all', ['coverage','classified_accuracy']].to_numpy(dtype=float)
        for k in (1,2):
            xy = part.loc[rules[k], ['coverage','classified_accuracy']].to_numpy(dtype=float)
            if np.isfinite(xy).all() and np.isfinite(base).all():
                ax.plot([base[0],xy[0]], [base[1],xy[1]], color=colors[k], alpha=.5, linewidth=1, zorder=1)
        offsets = [(0,-30),(-45,18),(0,-57)] if dataset in ('iris','wine') else [(0,-24),(12,7),(-32,-20)]
        for j, group in enumerate(groups):
            k = group['ids'][0]; x,y = group['xy']
            ax.scatter(x,y,s=120 if k==0 else 48,marker=markers[k],color=colors[k],
                       edgecolors='white',linewidths=.6,zorder=4 if k==0 else 3)
            text = 'All seven configurations' if len(group['ids'])==7 else ', '.join('R'+str(i) for i in group['ids'])
            offset = offsets[j] if j < len(offsets) else (20,-15)
            if dataset=='lenses' and j==1: offset=(10,-14)
            ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',
                ha='center' if offset[0]<=0 else 'left', va='center',fontsize=8,color=colors[k],
                arrowprops=dict(arrowstyle='-',color=colors[k],lw=.65,alpha=.7),
                bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.9))
        ax.set_title(f'({chr(97+index)}) {label}',loc='left',fontsize=10)
        ax.set_xlim(45,103);ax.set_ylim(65,103)
        ax.set_xticks([50,60,70,80,90,100]);ax.set_yticks([70,80,90,100])
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.2,linewidth=.5)
        if index%3==0:ax.set_ylabel('Classified accuracy (%)',fontsize=9)
        if index>=3:ax.set_xlabel('Coverage (%)',fontsize=9)
    key = axes.flat[5];key.axis('off')
    labels = ['R0  Full rule set',r'R1  $A_1,A_2$ only',r'R2  $B_1,B_2$ only',
              r'R3  Without $A_1$',r'R4  Without $A_2$',r'R5  Without $B_1$',r'R6  Without $B_2$']
    handles = [Line2D([],[],color=colors[i],marker=markers[i],linestyle='none',
                      markersize=10 if i==0 else 6,label=labels[i]) for i in range(7)]
    key.legend(handles=handles,loc='upper left',frameon=False,fontsize=8,labelspacing=.65)
    key.text(.03,.02,'Points: means across 3 seeds.\nCoincident means share one marker.',
             transform=key.transAxes,fontsize=8,color='#555555',va='bottom')
    fig.subplots_adjust(left=.08,right=.98,bottom=.10,top=.94,wspace=.28,hspace=.33)
    return fig


def plot_ablation_detail(data, *, datasets, rule_groups, metrics, figsize=(7.2, 3.7)):
    """One dataset, two absolute-value panels; independent, explicitly ticked ranges."""
    from matplotlib.ticker import MaxNLocator
    if len(datasets) != 1:
        raise ValueError('A detail figure requires exactly one dataset')
    dataset, title = next(iter(datasets.items()))
    labels = [r'$\{A_1,A_2,B_1,B_2\}$', r'$\{A_1,A_2\}$', r'$\{B_1,B_2\}$',
              r'$\{A_2,B_1,B_2\}$', r'$\{A_1,B_1,B_2\}$',
              r'$\{A_1,A_2,B_2\}$', r'$\{A_1,A_2,B_1\}$']
    expected = ['all','support_only','exclusion_only','without_A1','without_A2','without_B1','without_B2']
    if list(rule_groups) != expected:
        raise ValueError('Unexpected rule order in detail figure')
    colors = ['#222222','#0072B2','#D55E00','#009E73','#CC79A7','#B8860B','#56B4E9']
    fig, axes = plt.subplots(1,len(metrics),figsize=figsize,sharey=True,squeeze=False)
    for col,(metric,label) in enumerate(metrics.items()):
        ax=axes[0,col]
        part=data[data.dataset.eq(dataset)&data.metric.eq(metric)].set_index('rule_group').reindex(expected)
        mean=part.mean_pct.to_numpy(dtype=float); sd=part.sd_pct.to_numpy(dtype=float)
        valid=np.isfinite(mean)&np.isfinite(sd)
        if not valid.any(): raise ValueError(f'No defined values: {dataset}/{metric}')
        lo=float(np.min(mean[valid]-sd[valid])); hi=float(np.max(mean[valid]+sd[valid]))
        padding=max((hi-lo)*.15,.5)
        ax.set_xlim(0,100) if dataset == 'balloons' else ax.set_xlim(lo-padding,hi+padding)
        ax.axhspan(-.45,.45,color='#f1f3f5',zorder=0)
        for i in range(len(expected)):
            if valid[i]:
                ax.errorbar(mean[i],i,xerr=sd[i],fmt='D' if i==0 else 'o',
                            color=colors[i],markersize=5 if i==0 else 4.5,
                            capsize=3,elinewidth=1.25,markeredgecolor='white',markeredgewidth=.5,
                            clip_on=dataset != 'balloons',zorder=3)
            else:
                ax.text(.5,i,'Undefined',transform=ax.get_yaxis_transform(),ha='center',fontsize=8)
        ax.set_title(f'({chr(97+col)}) {label} (%)',loc='left',fontsize=10,pad=11)
        ax.set_ylim(6.6,-.6)
        ax.set_yticks(range(7),labels,fontsize=10)
        ax.tick_params(axis='y',length=0,pad=9)
        ax.tick_params(axis='x',labelsize=9,length=3,color='#888888')
        if dataset == 'balloons':
            ax.set_xticks([0,20,40,60,80,100])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.grid(axis='x',color='#e7e7e7',linewidth=.7,zorder=0)
        ax.axhline(2.5,color='#e3e3e3',linewidth=.8)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#aaaaaa')
    fig.text(.04,.96,title,fontsize=13,fontweight='bold',ha='left',va='top')
    fig.text(.97,.95,'Mean ± SD across 3 seeds',fontsize=8,color='#666666',ha='right',va='top')
    fig.subplots_adjust(left=.245,right=.97,top=.78,bottom=.13,wspace=.28)
    return fig


def plot_ablation_binary(data, *, datasets, rule_groups, metrics, figsize=(7.2, 3.6)):
    """Collapse only identical per-seed metrics; never average seven configurations as replicates."""
    seed_cols=[c for c in data.columns if c.startswith('seed_') and c.endswith('_pct')]
    if len(rule_groups)!=7 or len(seed_cols)!=3:
        raise ValueError('Binary summary requires seven configurations and three seeds')
    fig,axes=plt.subplots(1,len(datasets),figsize=figsize,sharey=True,squeeze=False)
    colors=['#0072B2','#D55E00']
    for col,(dataset,label) in enumerate(datasets.items()):
        ax=axes[0,col]
        means=[];stds=[]
        for metric in metrics:
            part=data[data.dataset.eq(dataset)&data.metric.eq(metric)].set_index('rule_group')
            if set(part.index)!=set(rule_groups) or len(part)!=len(rule_groups):
                raise ValueError(f'Missing or duplicate rule configuration: {dataset}/{metric}')
            values=part[seed_cols+['mean_pct','sd_pct']].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.allclose(values,values[0],rtol=0,atol=1e-10):
                raise ValueError(f'Cannot merge unequal configurations: {dataset}/{metric}')
            means.append(part.loc['all','mean_pct']);stds.append(part.loc['all','sd_pct'])
        x=np.arange(len(metrics))
        ax.bar(x,means,width=.52,color=colors[:len(metrics)],zorder=3)
        ax.errorbar(x,means,yerr=stds,fmt='none',ecolor='#333333',capsize=4,elinewidth=1,zorder=4)
        for i,(mean,sd) in enumerate(zip(means,stds)):
            ax.text(i,mean+sd+2,f'{mean:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set_title(f'({chr(97+col)}) {label}',loc='left',fontsize=11,pad=10)
        ax.set_xticks(x,['Classified\naccuracy','Coverage'],fontsize=10)
        ax.set_xlim(-.6,len(metrics)-.4);ax.set_ylim(0,112)
        ax.set_yticks([0,20,40,60,80,100]);ax.tick_params(axis='y',labelsize=9)
        ax.tick_params(axis='x',length=0,pad=8)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#aaaaaa')
        ax.grid(axis='y',color='#e5e5e5',linewidth=.7,zorder=0)
        if col==0:ax.set_ylabel('Performance (%)',fontsize=10)
    fig.text(.5,.97,'Identical metrics across all seven rule configurations',
             ha='center',va='top',fontsize=11)
    fig.text(.5,.9,'Mean ± SD across 3 seeds',ha='center',va='top',fontsize=8,color='#666666')
    fig.subplots_adjust(left=.09,right=.97,top=.75,bottom=.19,wspace=.28)
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
