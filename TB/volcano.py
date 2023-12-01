import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway, ttest_ind
from adjustText import adjust_text
from statsmodels.stats import multitest


class Volcano:
    def __init__(self, log2_transf=True, test_func=None):
        """
        Initialize the Volcano class.

        Parameters:
        - test_func: Statistical test function (e.g., f_oneway) for significance testing.
                     Defaults to T-test.
        """
        self.log2_transf = log2_transf

        if test_func is None:
            self.test_func = ttest_ind
        elif callable(test_func):
            self.test_func = test_func
        else:
            raise ValueError(f"{test_func} should be a callable function.")

        self.results = None
        self.group_labels = None
        self.significance_base_level = 0.05

    def fit(self, X, y, reference_state=None):
        """
        Perform a statistical test on the input data and calculate significance values.

        Parameters:
        - X: DataFrame with features.
        - y: List of labels for grouping.

        Returns:
        - DataFrame with calculated significance values, including p-values and fold changes.
        """
        data = pd.DataFrame(X)
        labels = list(y)

        features = data.columns.to_list()

        self.group_labels = self.get_group_labels(labels)

        if reference_state:
            label_0 = reference_state
            label_1 = [element for element in self.group_labels if element != reference_state][0]
        else:
            label_0, label_1 = self.group_labels

        print(f"Considering {label_0} as Reference State")

        data["labels"] = labels

        grps = data.groupby("labels")
        grp_0, grp_1 = grps.get_group(label_0), grps.get_group(label_1)

        p_values = []
        fold_changes = []

        results = pd.DataFrame()

        for feature in features:
            try:
                p_value = self.calculate_p_value(grp_0[feature], grp_1[feature])
                fold_change = self.calculate_fold_change(grp_0[feature], grp_1[feature])
                results.loc[feature, ["p-value", "fold-change"]] = p_value, fold_change
            except ZeroDivisionError as e:
                results.loc[feature, ["p-value", "fold-change"]] = 1, 1

        if self.log2_transf:
            results["log2(fold-change)"] = results["fold-change"]
        else:
            results["log2(fold-change)"] = results["fold-change"].apply(np.log2)

        results.index.name = "Feature"

        results = results.dropna(subset=["fold-change", "p-value"])

        results["p-value_corrected"] = self.pvalue_correction(results["p-value"])[1]
        results["-log10(p-value)"] = -results["p-value_corrected"].apply(np.log10)

        results["Significant"] = self.pvalue_correction(results["p-value"])[0]

        results = results[["fold-change", "log2(fold-change)", "p-value", "p-value_corrected",
                           "-log10(p-value)", "Significant"]]

        results = results.rename(columns={"fold-change": f"fold-change [{label_1} - {label_0}]",
                                          "log2(fold-change)": f"log2(fold-change) [{label_1} - {label_0}]"})

        self.results = results.reset_index()
        return self.results

    def get_group_labels(self, labels):
        """
        Get the unique group labels.

        Parameters:
        - labels: List of labels for grouping.

        Returns:
        - Tuple of unique group labels.
        """
        group_labels = list(set(labels))
        group_labels.sort()
        assert len(group_labels) == 2
        return group_labels[0], group_labels[1]

    def calculate_p_value(self, values_0, values_1):
        """
        Calculate the p-value using the specified test function.

        Parameters:
        - values_0: Values for group 0.
        - values_1: Values for group 1.

        Returns:
        - p-value.
        """
        if self.test_func == ttest_ind:
            return self.test_func(values_0, values_1, equal_var=False, nan_policy='omit').pvalue
        else:
            return self.test_func(values_0, values_1).pvalue

    def pvalue_correction(self, pvals):

        return multitest.fdrcorrection(pvals, alpha=self.significance_base_level, method='indep', is_sorted=False)

    def calculate_fold_change(self, values_0, values_1):
        """
        Calculate the fold change between two groups.

        Parameters:
        - values_0: Values for group 0.
        - values_1: Values for group 1.

        Returns:
        - Fold change.
        """
        if self.log2_transf:
            return np.mean(values_1) - np.mean(values_0)
        else:
            return np.mean(values_1) / np.mean(values_0)

    def plot_interactive(self, minfoldchange=1, highlight=None, height=750, width=750):
        """
        Create an interactive volcano plot.

        Parameters:
        - minfoldchange: Minimum fold change for significance.
        - highlight: List of features to highlight.
        - height: Plot height.
        - width: Plot width.

        Returns:
        - Plotly Figure for the interactive plot.
        """
        x = "log2(fold-change)"
        y = "-log10(p-value)"

        results = self.results.copy()

        results['color'] = 'a'

        results.loc[results.Significant & (results[x] > minfoldchange), 'color'] = 'b'
        results.loc[results.Significant & (results[x] < -minfoldchange), 'color'] = 'c'

        fig = px.scatter(
            data_frame=results,
            y=y,
            x=x,
            hover_data=["Feature", "p-value", "fold-change"],
            height=height,
            width=width,
            color="color",
        )

        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        fig.add_hline(
            y=-np.log10(self.significance_base_level), line_width=0.5, line_dash="dash"
        )

        fig.add_vline(x=minfoldchange, line_width=0.5, line_dash="dash")
        fig.add_vline(x=-minfoldchange, line_width=0.5, line_dash="dash")

        sig = self.results[self.results.Significant & (results[x].abs() > minfoldchange)]

        fig.add_trace(
            go.Scatter(
                x=sig[x],
                y=sig[y],
                mode="text",
                text=sig.Feature,
                name="Feature",
                textposition="top center",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(
                    [
                        results[x].min(),
                        results[x].max(),
                    ]
                ),
                y=np.array(
                    [results[y].max(), results[y].max()]
                )
                  * 1.1,
                mode="text",
                text=self.group_labels,
                name="Group",
                textposition="middle center",
                textfont=dict(color="crimson"),
                marker=dict(size=3),
                hoverinfo="skip",
            )
        )

        fig.update_layout(showlegend=False)

        return fig

    def plot(self, minfoldchange=1, nmaxannot=5, legend=False, highlight=None, **kwargs):
        """
        Create a static volcano plot.

        Parameters:
        - minfoldchange: Minimum fold change for significance.
        - nmaxannot: Maximum number of features to annotate.
        - legend: Show the legend.
        - highlight: List of features to highlight.
        - **kwargs: Additional keyword arguments for seaborn.scatterplot.

        Returns:
        - Matplotlib figure for the static plot.
        """
        x = "log2(fold-change)"
        y = "-log10(p-value)"

        results = self.results.copy()
        n_results = len(results)
        results["_colors_"] = get_colors(
            results.Significant, results[x], [minfoldchange] * n_results
        )

        g = sns.scatterplot(
            data=results, x=x, y=y, hue="_colors_", legend=legend, **kwargs
        )

        plt.axhline(
            y=-np.log10(self.significance_base_level), lw=0.5, ls="--", color="k"
        )

        plt.axvline(x=minfoldchange, lw=0.5, ls="--", color="k")
        plt.axvline(x=-minfoldchange, lw=0.5, ls="--", color="k")

        x_groups = [0.015, 0.985]
        y_groups = [0.03, 0.03]
        halign = ["left", "right"]

        ax = plt.gca()

        # Add group labels
        for i in [0, 1]:
            text = plt.text(
                x_groups[i],
                y_groups[i],
                self.group_labels[i],
                color=".3",
                horizontalalignment=halign[i],
                transform=ax.transAxes,
                backgroundcolor="0.3",
                bbox=dict(facecolor="none",
                          edgecolor="0.3",
                          boxstyle="round, pad=0.2"),
            )

        if highlight is None:
            annot = results[results.Significant & (results[x].abs() > minfoldchange)].sort_values("p-value")
            if nmaxannot is not None:
                annot = annot.head(nmaxannot)
        else:
            annot = results[results.Feature.isin(highlight)]

        # Add metabolite labels
        texts = []
        for ndx, row in annot.iterrows():
            x_value = row[x]
            y_value = row[y]
            _text = row["Feature"]

            if highlight is not None:
                plt.plot(x_value, y_value, mfc='none', mew=1, mec='cyan', marker='o')
            text = plt.text(
                x_value, y_value, _text, color="black", horizontalalignment="center"
            )
            texts.append(text)

        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="k", lw=0.5))

        sns.despine()
        return plt.gcf()


def _get_color_(sig, log_fc, minfoldchange=1):
    """
    Get colors for points on the volcano plot.

    Parameters:
    - sig: Significance of the point.
    - log_fc: Log2 fold change of the point.
    - minfoldchange: Minimum fold change for significance.

    Returns:
    - Color for the point.
    """
    if sig:
        if log_fc >= minfoldchange:
            return "C2"
        elif log_fc <= -minfoldchange:
            return "C3"
    return "C1"


get_colors = np.vectorize(_get_color_)
