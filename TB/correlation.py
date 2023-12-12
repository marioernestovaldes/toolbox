import pandas as pd
import numpy as np
import scipy
from statsmodels.stats import multitest
import logging


class DiffNetworkAnalysis:
    """
    A class for performing differential correlation analysis between two states (stateA and stateB).

    Parameters:
    - stateA (pd.DataFrame): Data for stateA.
    - stateB (pd.DataFrame): Data for stateB.
    - correlation (str): Type of correlation to be used ('pearson' or 'spearman').
    - significance_base_level (float): Base level of significance for hypothesis testing.

    Example:
    analysis = DiffNetworkAnalysis(stateA=df_stateA, stateB=df_stateB, correlation='pearson', significance_base_level=0.01)
    corr_stateA, corr_stateB, corr_diff = analysis.diff_corr_network()
    """

    def __init__(self, stateA: pd.DataFrame = None, stateB: pd.DataFrame = None, correlation: str = 'pearson',
                 significance_base_level: float = 0.01):
        if correlation not in ['pearson', 'spearman']:
            raise ValueError(f"{correlation} should be 'pearson' or 'spearman'.")
        self.stateA, self.stateB = stateA, stateB
        self.correlation = correlation
        self.significance_base_level = significance_base_level

    def diff_corr_network(self):
        """
        Perform differential correlation analysis between stateA and stateB.

        Returns:
        - df_r_stateA (pd.DataFrame): Correlation network for stateA.
        - df_r_stateB (pd.DataFrame): Correlation network for stateB.
        - df_r_diff (pd.DataFrame): Differential correlation results between stateA and stateB.
        """
        logging.info('Processing stateA and stateB dataframes...')
        df_r_stateA, df_r_stateB = self.corr_network(self.stateA), self.corr_network(self.stateB)

        logging.info('Generating differential correlation for states A and B...')
        df_r_diff = pd.concat([df_r_stateA[['Prot1', 'Prot2']],
                               df_r_stateA['r-value'] - df_r_stateB['r-value']], axis=1)

        df_r_diff['abs_diff'] = df_r_diff['r-value'].abs()

        df_r_diff = (
            df_r_diff
            .sort_values(by='abs_diff', ascending=False)
            .reset_index(drop=True)
            .drop('abs_diff', axis=1).
            rename(columns={'r-value': 'Corr_diff'})
        )

        return df_r_stateA, df_r_stateB, df_r_diff

    def corr_network(self, df: pd.DataFrame):
        """
        Generate a correlation network for the given DataFrame.

        Parameters:
        - df (pd.DataFrame): Data for certain state.

        Returns:
        - df_r (pd.DataFrame): Correlation data.
        """
        df_r = df.corr(method=self.correlation)
        df_r = df_r.unstack().reset_index()

        n_samples = df.shape[0]
        n_features = df.shape[1]

        # calculate p-values
        pvalues = self.derive_pvalues(df_r[0], n_samples)

        # multiple hypothesis correction
        pvalues_corrected = self.correct_pvalues(pvalues)

        df_r.columns = ['Prot1', 'Prot2', 'r-value']

        df_r['r-value_abs'] = df_r['r-value'].abs()
        df_r['p-value_corrected'] = pvalues_corrected[1]
        df_r[f'hypothesis rejected for alpha = {self.significance_base_level}'] = pvalues_corrected[0]

        df_r['r-value'] = df_r.apply(lambda x: np.nan if x['Prot1'] == x['Prot2'] else x['r-value'], axis=1)

        return df_r

    def derive_pvalues(self, correlations, n_samples):
        """
        Calculate p-values from correlations given the number of samples used to calculate the correlations.

        Parameters:
        - correlations: Correlation values.
        - n_samples: Number of samples.

        Returns:
        - pvals: Calculated p-values.
        """
        correlations = np.asarray(correlations)
        rf = correlations
        df = n_samples - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pvals = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
        return pvals

    def correct_pvalues(self, pvals):
        """
        Correct p-values for multiple hypothesis testing.

        Parameters:
        - pvals: Original p-values.

        Returns:
        - pvals_corrected: Corrected p-values.
        """
        return multitest.fdrcorrection(pvals, alpha=self.significance_base_level, method='indep', is_sorted=False)


