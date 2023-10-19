import numpy as np
import seaborn as sns
import matplotlib as mpl
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import norm
from pathlib import Path as P
from adjustText import adjust_text

def labeled_scatterplot(df, x, y, label, **kwargs):
    """
    Creates a labeled scatter plot.

    Parameters:
    - df: DataFrame containing the data.
    - x: Column name for the x-axis.
    - y: Column name for the y-axis.
    - label: Column name for labels to be displayed.
    - **kwargs: Additional keyword arguments for Seaborn scatterplot.

    Returns:
    - The scatter plot with labels.
    """
    sns.scatterplot(x=df[x], y=df[y], **kwargs)
    texts = []
    for ndx, row in df.iterrows():
        x_value = row[x]
        y_value = row[y]
        _text = row[label]
        text = plt.text(x_value, y_value, _text, color="black", horizontalalignment="center")
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="k", lw=0.5))
    return plt.gcf()

def plot_roc(target, score, cutoff_target=None, ax=None, pos_label=None, add_text=False,
             set_tick_labels=True, estimate_random=True, with_auc=True, **kwargs):
    """
    Plots a Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters:
    - target: True binary labels.
    - score: Target scores.
    - cutoff_target: A threshold to classify the target (default: None).
    - ax: Matplotlib axis object (default: None).
    - pos_label: The positive label (default: None).
    - add_text: Whether to add AUC text to the plot (default: False).
    - set_tick_labels: Whether to set tick labels (default: True).
    - estimate_random: Whether to plot random ROC curves for comparison (default: True).
    - with_auc: Whether to compute and return the AUC (default: True).
    - **kwargs: Additional keyword arguments for Matplotlib plot function.

    Returns:
    - The AUC value if `with_auc` is True, otherwise, the ROC curve is plotted.
    """
    ax = _activate_axis_(ax)
    if cutoff_target is not None:
        target = classify(target, cutoff_target)
    fpr, tpr, _ = roc_curve(target, score, pos_label=pos_label)
    auc = roc_auc_score(target, score)
    plt.plot(fpr, tpr, **kwargs)
    if add_text:
        plt.text(0.75, 0.04, f"AUC={auc:4.2f}", size=8)
    if estimate_random:
        plot_random_roc(target, 200, ax=ax)
    _plot_roc_defaults_(set_tick_labels=set_tick_labels, ax=ax)
    if with_auc:
        return auc

def _classify(i, value, inverse=False):
    """
    Helper function for classification. Classifies values based on a reference value.

    Parameters:
    - i: Integer value to classify.
    - value: Reference value for classification.
    - inverse: True to invert the classification.

    Returns:
    - True when i <= the reference value (or False when inverse=True), otherwise False (or True when inverse=True).
    """
    if inverse is False:
        return i > value
    else:
        return i <= value

classify = np.vectorize(_classify)

def _plot_roc_defaults_(set_tick_labels=True, ax=None, roc_percent=True):
    """
    Sets default settings for ROC plots.

    Parameters:
    - set_tick_labels: Whether to set tick labels (default: True).
    - ax: Matplotlib axis object (default: None).
    - roc_percent: Whether to use percentage labels on the ROC plot (default: True).
    """
    ax = _activate_axis_(ax)
    if set_tick_labels is False:
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
    else:
        if roc_percent:
            plt.xticks([0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80])
            plt.yticks([0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80])
            plt.xlabel("False Positive [%]")
            plt.ylabel("True Positive [%]")
        else:
            plt.xticks([0.2, 0.4, 0.6, 0.8])
            plt.yticks([0.2, 0.4, 0.6, 0.8])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plot_diagonal(linestyle="--", color="w")
    return ax

def plot_random_roc(labels, N, ax=None):
    """
    Generates and overlays random ROC curves for comparison.

    Parameters:
    - labels: True binary labels.
    - N: Number of random ROC curves to generate.
    - ax: Matplotlib axis object (default: None).

    Returns:
    - The axis with random ROC curves overlaid.
    """
    ax = _activate_axis_(ax)
    for i in range(N):
        plt.plot(*_random_roc_(labels), alpha=0.01, linewidth=10, color="k", zorder=0)
    return ax

def plot_diagonal(ax=None, **kwargs):
    """
    Plots a diagonal line on the graph.

    Parameters:
    - ax: Matplotlib axis object (default: None).
    - **kwargs: Additional keyword arguments for Matplotlib plot function.
    """
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    plt.plot([np.min([x0, y0]), np.min([x1, y1])], [np.min([x0, y0]), np.min([x1, y1])], **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))

def plot_hlines(hlines=None, ax=None, color=None, label=None, **kwargs):
    """
    Plots horizontal lines on the graph.

    Parameters:
    - hlines: List of horizontal line positions.
    - ax: Matplotlib axis object (default: None).
    - color: Line color.
    - label: Label for the lines.
    - **kwargs: Additional keyword arguments for Matplotlib hlines function.
    """
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if hlines is not None:
        if not isinstance(hlines, list):
            hlines = [hlines]
        for i, hline in enumerate(hlines):
            plt.hlines(hline, x0 - 0.2, x1 + 1.2, color=color, label=label if i == 0 else None, **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))

def plot_vlines(vlines=None, ax=None, color=None, label=None, **kwargs):
    """
    Plots vertical lines on the graph.

    Parameters:
    - vlines: List of vertical line positions.
    - ax: Matplotlib axis object (default: None).
    - color: Line color.
    - label: Label for the lines.
    - **kwargs: Additional keyword arguments for Matplotlib vlines function.
    """
    ax = _activate_axis_(ax)
    x0, x1, y0, y1 = _axis_dimensions_(ax)
    if vlines is not None:
        if not isinstance(vlines, list):
            vlines = [vlines]
        for i, vline in enumerate(vlines):
            plt.vlines(vline, y0 - 0.2, y1 + 1.2, color=color, label=label if i == 0 else None, **kwargs)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))

def _random_roc_(y_train, ax=None):
    """
    Generates random prediction and returns fpr, tpr used to plot a ROC-curve.

    Parameters:
    - y_train: True binary labels.
    - ax: Matplotlib axis object (default: None).

    Returns:
    - The false positive rate (fpr) and true positive rate (tpr) for the random ROC curve.
    """
    rand_prob = norm.rvs(size=len(y_train))
    fpr, tpr, _ = roc_curve(y_train, rand_prob, pos_label=1)
    return fpr, tpr

def _activate_axis_(ax=None):
    """
    Activates the given axis or uses the current axis.

    Parameters:
    - ax: Matplotlib axis object (default: None).

    Returns:
    - The activated axis.
    """
    if ax is not None:
        plt.sca(ax)
    return plt.gca()

def _axis_dimensions_(ax=None):
    """
    Returns the dimensions of a given axis.

    Parameters:
    - ax: Matplotlib axis object (default: None).

    Returns:
    - Tuple of x0, x1, y0, y1 representing the axis dimensions.
    """
    ax = _activate_axis_(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x0, x1, y0, y1)

def heatmap(dm, vmin=0, vmax=1):
    """
    Generates a heatmap with hierarchical clustering of the given data matrix.

    Parameters:
    - dm: Data matrix for the heatmap.
    - vmin: Minimum value for color mapping (default: 0).
    - vmax: Maximum value for color mapping (default: 1).

    Returns:
    - A heatmap object with hierarchical clustering.
    """
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform

    D1 = squareform(pdist(dm, metric="euclidean"))
    D2 = squareform(pdist(dm.T, metric="euclidean"))
    f = plt.figure(figsize=(8, 8))
    ax1 = f.add_axes([0.09, 0.1, 0.2, 0.6])
    Y = linkage(D1, method="complete")
    Z1 = dendrogram(Y, orientation="left")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = f.add_axes([0.3, 0.71, 0.6, 0.2])
    Y = linkage(D2, method="complete")
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    axmatrix = f.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1["leaves"]
    idx2 = Z2["leaves"]
    D = dm[idx1, :]
    D = D[:, idx2]
    axmatrix.matshow(D[::-1], aspect="auto", cmap="hot", vmin=vmin, vmax=vmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    return {"ordered": D, "rorder": Z1["leaves"], "corder": Z2["leaves"]}

def legend_outside(ax=None, bbox_to_anchor=None, **kwargs):
    """
    Places the legend outside the current axis.

    Parameters:
    - ax: Matplotlib axis object (default: None).
    - bbox_to_anchor: 2D tuple with numeric values (default: (1, 1.05)).
    - **kwargs: Additional keyword arguments for Matplotlib legend function.
    """
    if ax is None:
        ax = plt.gca()
    if bbox_to_anchor is None:
        bbox_to_anchor = (1, 1.05)
    ax.legend(bbox_to_anchor=bbox_to_anchor, **kwargs)

def plot_dendrogram(df, labels=None, orientation="left", metric=None, color_threshold=0,
                    above_threshold_color="k", **kwargs):
    """
    Plots a dendrogram based on hierarchical clustering.

    Parameters:
    - df: Data for hierarchical clustering.
    - labels: Labels for dendrogram leaves (default: None).
    - orientation: Orientation of the dendrogram ("left" or "top", default: "left").
    - metric: The distance metric for clustering (default: None).
    - color_threshold: Threshold for coloring branches (default: 0).
    - above_threshold_color: Color for branches above the threshold (default: "k").
    - **kwargs: Additional keyword arguments for scipy hierarchy.dendrogram function.

    Returns:
    - The hierarchical linkage matrix and the dendrogram tree.
    """
    Z = hierarchy.linkage(df, metric=metric)
    T = hierarchy.to_tree(Z)
    data = hierarchy.dendrogram(
        Z, labels=labels, orientation=orientation, color_threshold=color_threshold,
        above_threshold_color=above_threshold_color, **kwargs
    )
    ndx = data["leaves"]
    if orientation in ["left", "right"]:
        plt.xticks([])
    if orientation in ["top", "bottom"]:
        plt.xticks([])
    plt.gca().set(frame_on=False)
    return Z, T

def savefig(name, fmt=["pdf", "png", "svg"], bbox_inches="tight", dpi=300):
    """
    Saves the current figure with various file formats and settings.

    Parameters:
    - name: Name for the saved figure.
    - notebook_name: Name of the notebook (default: None).
    - fmt: List of file formats to save (default: ["pdf", "png", "svg"]).
    - bbox_inches: Bounding box in inches (default: "tight").
    - dpi: Dots per inch (default: 300).
    """
    fig = plt.gcf()
    for suffix in fmt:
        fig.savefig(f'{name}.{suffix}', bbox_inches=bbox_inches, dpi=dpi)
        print(f"Saved: {name}.{suffix}")
