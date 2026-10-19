"""四类绘图函数。只绘图，不读文件、不筛选实验、不计算实验指标。"""
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
