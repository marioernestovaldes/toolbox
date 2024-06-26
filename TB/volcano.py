import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from adjustText import adjust_text
from statsmodels.stats import multitest
import sys


class Volcano:
    def __init__(self, log2_transf=True, test_func=mannwhitneyu, significance_base_level=0.05):
        """
        Initialize the Volcano class.

        Parameters:
        - test_func: Statistical test function (e.g., mannwhitneyu) for significance testing.
                     Defaults to Mann-Whitney U rank test on two independent samples.
        """
        self.log2_transf = log2_transf

        if callable(test_func):
            self.test_func = test_func
        else:
            raise ValueError(f"{test_func} should be a callable function.")

        self.results = None
        self.group_labels = None
        self.stateA = None
        self.stateB = None
        self.significance_base_level = significance_base_level

    def fit(self, df: pd.DataFrame, y: list, reference_state=None):
        """
        Perform a statistical test on the input data and calculate significance values.

        Parameters:
        - df: DataFrame with features.
        - y: List of labels for grouping.

        Returns:
        - DataFrame with calculated significance values, including p-values and fold changes.
        """
        data = pd.DataFrame(df)
        labels = list(y)

        features = data.columns.to_list()

        self.setup_groups(labels, reference_state)

        data["labels"] = labels

        grps = data.groupby("labels")
        grp_0, grp_1 = grps.get_group(self.stateA), grps.get_group(self.stateB)

        results = self.calculate_feature_statistics(data.drop("labels", axis=1).columns.to_list(), grp_0, grp_1)

        results = self.transform_and_correct_results(results)

        self.results = results.reset_index().sort_values(by=[f"log2(fold-change) [{self.stateB} - {self.stateA}]",
                                                             "-log10(p-value)"],
                                                         ascending=False).reset_index(drop=True)
        return self.results

    def setup_groups(self, labels, reference_state):
        self.group_labels = self.get_group_labels(labels)

        if reference_state:
            stateA = reference_state
            stateB = [element for element in self.group_labels if element != reference_state][0]
        else:
            stateA, stateB = self.group_labels

        self.stateA, self.stateB = stateA, stateB
        print(f"Considering {self.stateA} as Reference State and using {self.test_func} for significance testing...")

    def calculate_feature_statistics(self, features, grp_0, grp_1):
        results = pd.DataFrame()

        for feature in features:
            try:
                p_value = self.calculate_p_value(grp_0[feature], grp_1[feature])

                p_value = sys.float_info.min if p_value == 0 else p_value

                fold_change = self.calculate_fold_change(grp_0[feature], grp_1[feature])
                results.loc[feature, ["p-value", "fold-change"]] = p_value, fold_change
            except Exception as e:
                print(e)
                print(f"Exception for {feature}... assigning p-value, fold-change = 1, 0 respectively")
                results.loc[feature, ["p-value", "fold-change"]] = 1, 0

        return results

    def transform_and_correct_results(self, results):
        if self.log2_transf:
            results["log2(fold-change)"] = results["fold-change"]
        else:
            results["log2(fold-change)"] = results["fold-change"].apply(np.log2)

        results.index.name = "Feature"
        results = results.dropna(subset=["fold-change", "p-value"])

        results["p-value_corrected"] = self.pvalue_correction(results["p-value"])[1]
        results["-log10(p-value)"] = -results["p-value_corrected"].apply(np.log10)
        results["Significant"] = self.pvalue_correction(results["p-value"])[0]

        results = results[
            ["fold-change", "log2(fold-change)", "p-value", "p-value_corrected", "-log10(p-value)", "Significant"]]
        results = results.rename(columns={"fold-change": f"fold-change [{self.stateB} - {self.stateA}]",
                                          "log2(fold-change)": f"log2(fold-change) [{self.stateB} - {self.stateA}]"})

        return results

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
        if self.test_func == mannwhitneyu:
            return self.test_func(values_0, values_1, nan_policy='omit').pvalue
        elif self.test_func == ttest_ind:
            return self.test_func(values_0, values_1, equal_var=False, nan_policy='omit').pvalue
        else:
            return self.test_func(values_0, values_1).pvalue

    def pvalue_correction(self, pvals):

        return multitest.fdrcorrection(pvals, alpha=self.significance_base_level, method='indep', is_sorted=False)

    def select_ratio(self, values_0, values_1, is_log2_transf=True):
        if is_log2_transf:
            return values_1 - values_0
        else:
            return values_1 / values_0

    def calculate_fold_change(self, values_0, values_1):
        """
        Calculate the fold change between two groups.

        Parameters:
        - values_0: Values for group 0.
        - values_1: Values for group 1.

        Returns:
        - Fold change.
        """
        if self.test_func == mannwhitneyu:
            return self.select_ratio(np.nanmedian(values_0), np.nanmedian(values_1), is_log2_transf=self.log2_transf)
        else:
            return self.select_ratio(np.nanmean(values_0), np.nanmean(values_1), is_log2_transf=self.log2_transf)

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
        x = f"log2(fold-change) [{self.stateB} - {self.stateA}]"
        y = "-log10(p-value)"

        results = self.results.copy()

        results['color'] = 'silver'

        results.loc[results.Significant & (results[x] > minfoldchange), 'color'] = 'b'
        results.loc[results.Significant & (results[x] < -minfoldchange), 'color'] = 'r'

        fig = px.scatter(
            data_frame=results,
            y=y,
            x=x,
            hover_data=["Feature", "p-value", f"log2(fold-change) [{self.stateB} - {self.stateA}]"],
            height=height,
            width=width,
            color='color',
        )

        fig.update_traces(
            marker=dict(size=7, line=dict(width=1, color="DarkSlateGrey")),
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
                text=[self.stateA, self.stateB],
                name="Group",
                textposition="top center",
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
        x = f"log2(fold-change) [{self.stateB} - {self.stateA}]"
        y = "-log10(p-value)"

        plt.clf()

        results = self.results.copy()

        fig, ax = plt.subplots()

        g = sns.scatterplot(
            data=results, x=x, y=y, c=[_get_color_(log10_p, log_fc) for log10_p, log_fc in zip(results[y].to_list(),
                                                                                               results[x].to_list())],
            legend=legend, ax=ax, **kwargs
        )

        g.set_xlim(-(results[x].max()), results[x].max())

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
                [self.stateA, self.stateB][i],
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

        # Add labels
        texts = []
        for ndx, row in annot.iterrows():
            x_value = row[x]
            y_value = row[y]
            _text = row["Feature"]

            # if highlight is not None:
            #     plt.plot(x_value, y_value, mfc='none', mew=1, mec='cyan', marker='o')
            text = plt.text(
                x_value, y_value, _text, color="black", horizontalalignment="center"
            )
            texts.append(text)

        if len(texts) > 0:
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="k", lw=0.5))

        sns.despine()

        return fig


def _get_color_(log10_p, log_fc, minfoldchange=1):
    """
    Get colors for points on the volcano plot.

    Parameters:
    - sig: Significance of the point.
    - log_fc: Log2 fold change of the point.
    - minfoldchange: Minimum fold change for significance.

    Returns:
    - Color for the point.
    """
    if log_fc >= minfoldchange and log10_p >= 1.3:
        return "#EE7733"
    elif log_fc <= -minfoldchange and log10_p >= 1.3:
        return "#009988"
    else:
        return "#BBBBBB"
