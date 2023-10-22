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

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, size, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            # ax.text(*self.bubbles[i, :2], labels[i]+f'\n({count[i]})',
            #        horizontalalignment='center', verticalalignment='center')
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='bottom', fontsize=12)
            ax.text(*self.bubbles[i, :2], f'({size[i]})',
                    horizontalalignment='center', verticalalignment='top', fontsize=9)

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
