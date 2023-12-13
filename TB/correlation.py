import logging
import numpy as np
import pandas as pd
import scipy.special
from statsmodels.stats import multitest


class DiffNetworkAnalysis:
    """
    A class for performing differential correlation analysis between two states (stateA and stateB).

    Parameters:
    - stateA (pd.DataFrame): Data for stateA.
    - stateB (pd.DataFrame): Data for stateB.
    - correlation (str): Type of correlation to be used ('pearson' or 'spearman').
    - significance_base_level (float): Base level of significance for hypothesis testing (between 0 and 1).

    Example:
    analysis = DiffNetworkAnalysis(stateA=df_stateA, stateB=df_stateB, correlation='pearson', significance_base_level=0.01)
    corr_stateA, corr_stateB, corr_diff = analysis.diff_corr_network()
    """

    def __init__(self, stateA: pd.DataFrame = None, stateB: pd.DataFrame = None, correlation: str = 'pearson',
                 significance_base_level: float = 0.01):
        """
        Constructor for DiffNetworkAnalysis class.

        Parameters:
        - stateA (pd.DataFrame): Data for stateA.
        - stateB (pd.DataFrame): Data for stateB.
        - correlation (str): Type of correlation to be used ('pearson' or 'spearman').
        - significance_base_level (float): Base level of significance for hypothesis testing (between 0 and 1).
        """
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
        df_r_diff = pd.concat(
            [
                df_r_stateA[['Prot1', 'Prot2']],
                df_r_stateA['r-value'] - df_r_stateB['r-value'],
                (df_r_stateA[f'hypothesis rejected for alpha = {self.significance_base_level}'] &
                 df_r_stateA[f'hypothesis rejected for alpha = {self.significance_base_level}'])
            ], axis=1)

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
        df_r = df_r.unstack()

        # pairs = df_r.index.to_list()
        #
        # # Count matching non-NaN values for each pair using numpy
        # matching_counts = []
        #
        # for col1, col2 in pairs:
        #     matching_count = np.sum(~np.isnan(df[col1]) & ~np.isnan(df[col2]))
        #     matching_counts.append({'Prot1': col1, 'Prot2': col2, 'MatchingNonNaNCount': matching_count})

        # Create a DataFrame with the results
        n_samples_df = self.pairwise_non_nan_values(df)

        # calculate p-values
        pvalues = self.derive_pvalues(df_r.to_list(), n_samples_df['MatchingNonNaNCount'])

        # multiple hypothesis correction
        pvalues_corrected = self.correct_pvalues(pvalues)

        df_r = df_r.reset_index()

        df_r.columns = ['Prot1', 'Prot2', 'r-value']

        df_r['r-value_abs'] = df_r['r-value'].abs()
        df_r['p-value_corrected'] = pvalues_corrected[1]
        df_r[f'hypothesis rejected for alpha = {self.significance_base_level}'] = pvalues_corrected[0]

        df_r['r-value'] = df_r.apply(lambda x: np.nan if x['Prot1'] == x['Prot2'] else x['r-value'], axis=1)
        df_r['r-value_abs'] = df_r.apply(lambda x: np.nan if x['Prot1'] == x['Prot2'] else x['r-value'], axis=1)

        df_r = pd.merge(df_r, n_samples_df, on=['Prot1', 'Prot2'])

        return df_r

    def pairwise_non_nan_values(self, df: pd.DataFrame):
        """
        Calculate the pairwise count of non-NaN values for each column combination in a DataFrame.

        This function iterates through each pair of columns in the provided DataFrame,
        calculating the count of non-NaN entries that both columns share. This is useful for
        understanding the overlap of non-missing data between different features in the dataset.

        Parameters:
        df (pd.DataFrame): A pandas DataFrame with any number of columns.

        Returns:
        pd.DataFrame: A DataFrame with three columns - 'Prot1', 'Prot2', and 'MatchingNonNaNCount'.
                      'Prot1' and 'Prot2' represent the column pairs, and 'MatchingNonNaNCount'
                      is the count of non-NaN values shared between these columns.
        """
        cols = df.columns

        # Convert the DataFrame to a numpy array, treating NaNs as np.nan
        mat = df.to_numpy(dtype=float, na_value=np.nan, copy=False).T

        # Create a mask that identifies where non-NaN values are located
        mask = np.isfinite(mat)

        # Initialize a list to store the pairwise non-NaN value counts
        pairwise_non_nan_values = []

        # Iterate through all pairs of columns
        for i, col_i in zip(range(0, df.shape[1]), cols):
            for j, col_j in zip(range(0, df.shape[1]), cols):
                # Count the number of positions where both columns have non-NaN values
                non_nan_count = (mask[i] & mask[j]).sum()
                pairwise_non_nan_values.append([col_i, col_j, non_nan_count])

        # Return the results as a DataFrame
        return pd.DataFrame(pairwise_non_nan_values, columns=['Prot1', 'Prot2', 'MatchingNonNaNCount'])

    def derive_pvalues(self, correlations, n_samples):
        """
        Calculate p-values from correlations given the number of samples used to calculate the correlations.

        Parameters:
        - correlations: Correlation values.
        - n_samples: Number of samples.

        Returns:
        - pvals: Calculated p-values.
        """
        # Define a constant for avoiding magic numbers
        NA_CORRECTION = 0.9999

        pvals = []
        for correlation, n_sample in zip(correlations, n_samples):
            rf = NA_CORRECTION if correlation == 1 else correlation
            df = n_sample - 2
            ts = rf * rf * (df / (1 - rf * rf))

            pval = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
            pvals.append(pval)

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
