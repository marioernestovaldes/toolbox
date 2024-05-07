import logging
import numpy as np
import pandas as pd
import scipy.special
from statsmodels.stats import multitest
from scipy.stats import binomtest


class DiffNetworkAnalysis:
    """
    A class for performing differential correlation analysis between two states (stateA and stateB).

    This class calculates the differential correlation network for two different states of a biological or chemical
    system. It involves calculating correlation networks for each state and then determining the differences between
    these networks. The class also includes methods for handling NaN values and correcting for multiple hypothesis
    testing.

    Parameters:
    - stateA (pd.DataFrame): Data for stateA.
    - label_A (str): Label for stateA.
    - stateB (pd.DataFrame): Data for stateB.
    - label_B (str): Label for stateB.
    - correlation (str): Type of correlation to be used ('pearson' or 'spearman').
    - significance_base_level (float): Base level of significance for hypothesis testing (between 0 and 1).

    Example:
    analysis = DiffNetworkAnalysis(stateA=df_stateA, label_A='State A',
                                   stateB=df_stateB, label_B='State B',
                                   correlation='pearson', significance_base_level=0.01)
    corr_stateA, corr_stateB, corr_diff = analysis.diff_corr_network()
    """

    def __init__(self,
                 stateA: pd.DataFrame = None, label_A: str = None, stateA_correlation: str = 'pearson',
                 stateB: pd.DataFrame = None, label_B: str = None, stateB_correlation: str = 'pearson',
                 replace_zeros: bool = True,
                 significance_base_level: float = 0.01):
        """
        Constructor for DiffNetworkAnalysis class.

        Initializes the class with the provided states, labels, correlation type, and significance level.
        It also validates the correlation type to ensure it is either 'pearson' or 'spearman'.
        """
        self.stateA, self.stateB = stateA, stateB
        self.label_A, self.label_B = label_A, label_B
        self.stateA_correlation = stateA_correlation
        self.stateB_correlation = stateB_correlation
        self.replace_zeros = replace_zeros
        self.significance_base_level = significance_base_level

    def diff_corr_network(self):
        """
        Perform differential correlation analysis between stateA and stateB.

        This method computes the correlation networks for each state and then calculates
        the differences between these networks. It also filters based on the significance of the differences.

        Returns:
        - df_r_stateA (pd.DataFrame): Correlation network for stateA.
        - df_r_stateB (pd.DataFrame): Correlation network for stateB.
        - df_r_diff (pd.DataFrame): Differential correlation results between stateA and stateB.
        """
        print(f'Processing stateA and stateB dataframes with replace_zeros={self.replace_zeros}')
        print(f"Using {self.stateA_correlation} for {self.label_A}...")
        df_r_stateA = self.corr_network(self.stateA, label=self.label_A, correlation=self.stateA_correlation,
                                        replace_zeros=self.replace_zeros)

        print(f"Using {self.stateB_correlation} for {self.label_B}...")
        df_r_stateB = self.corr_network(self.stateB, label=self.label_B, correlation=self.stateB_correlation,
                                        replace_zeros=self.replace_zeros)

        print('Generating differential correlation for states A and B...')

        df_r_diff = pd.merge(df_r_stateA, df_r_stateB,
                             on=['Prot1', 'Prot2', 'Prot_pair'], how='outer',
                             suffixes=(f'_{self.label_A}', f'_{self.label_B}'))

        df_r_diff[f'hypothesis rejected for alpha = {self.significance_base_level}'] = (
                    df_r_diff[f'hypothesis rejected for alpha = {self.significance_base_level}_{self.label_A}'] &
                    df_r_diff[f'hypothesis rejected for alpha = {self.significance_base_level}_{self.label_B}']
        )
        df_r_diff['Corr_diff'] = df_r_diff[f'r-value_{self.label_A}'] - df_r_diff[f'r-value_{self.label_B}']

        df_r_diff['abs_diff'] = df_r_diff['Corr_diff'].abs()

        df_r_diff = (
            df_r_diff
            .sort_values(by='abs_diff', ascending=False)
            .reset_index(drop=True)
            .drop('abs_diff', axis=1)
        )

        df_r_diff = df_r_diff.dropna(
            subset=[f'r-value_{self.label_A}', f'r-value_{self.label_B}', 'Corr_diff']).reset_index(drop=True)

        return df_r_stateA, df_r_stateB, df_r_diff

    def corr_network(self, df: pd.DataFrame, label: str=None, correlation='pearson', replace_zeros=True):
        """
        Generate a correlation network for the given DataFrame.

        This method calculates the correlation matrix for the provided DataFrame and
        transforms it into a format suitable for network analyses. It also handles NaN values
        and computes p-values and their corrections.

        Parameters:
        - df (pd.DataFrame): Data for a certain state.

        Returns:
        - df_r (pd.DataFrame): Correlation data with additional statistics.
        """
        if replace_zeros and not self.is_dataframe_binary_with_nan(df):
            df = df.replace(0, np.nan)

        df_r = df.corr(method=correlation).astype(float)

        # Set the upper triangle of the symmetric dataframe to NaN
        upper_triangle_indices = np.triu_indices(n=df_r.shape[0], k=0)  # Get the indices for the upper triangle
        df_r.values[upper_triangle_indices] = np.nan  # Set the upper triangle values to NaN

        # Transform the pairwise matrix into a dataframe suitable for network analyses
        df_r = (
            df_r
            .unstack()
            .reset_index()
            .rename(columns={'level_0': 'Prot1', 'level_1': 'Prot2', 0: 'r-value'})
        )

        df_r['Prot_pair'] = df_r['Prot1'] + '-' + df_r['Prot2']

        # Create a DataFrame with the matching_nonNaN_count per protein pair
        n_samples_df = self.pairwise_non_nan_values(df)

        df_r = pd.merge(n_samples_df, df_r, on=['Prot1', 'Prot2'])

        df_r = df_r.dropna(axis=0).reset_index(drop=True)

        df_r = df_r[['Prot1', 'Prot2', 'Prot_pair', 'matching_nonNaN_count', 'r-value']]

        # Calculate p-values and use multiple hypothesis correction
        if correlation == 'accuracy_score':
            pvalues = self.derive_pvalues_accuracies(df_r['r-value'], df_r['matching_nonNaN_count'], df)
        else:
            pvalues = self.derive_pvalues(df_r['r-value'], df_r['matching_nonNaN_count'])

        pvalues_corrected = self.correct_pvalues(pvalues)

        df_r['r-value_abs'] = df_r['r-value'].abs()
        df_r['p-value'] = pvalues
        df_r['p-value_corrected'] = pvalues_corrected[1]
        df_r[f'hypothesis rejected for alpha = {self.significance_base_level}'] = pvalues_corrected[0]

        if label:
            df_r['label'] = label

        return df_r

    def is_dataframe_binary_with_nan(self, df: pd.DataFrame):
        """
        Check if a pandas DataFrame contains only binary data (0 and 1) and possibly NaN values in all columns.

        Parameters:
        - df: pandas DataFrame to check.

        Returns:
        - True if all columns in the DataFrame contain only 0, 1, and possibly NaN, False otherwise.
        """
        for col in df.columns:
            if not set(df[col].dropna().unique()).issubset({0, 1}):
                return False
        return True

    def pairwise_non_nan_values(self, df: pd.DataFrame):
        """
        Calculate the pairwise count of non-NaN values for each column combination in a DataFrame.

        This function iterates through each pair of columns in the provided DataFrame,
        calculating the count of non-NaN entries that both columns share. This is useful for
        understanding the overlap of non-missing data between different features in the dataset.

        Parameters:
        df (pd.DataFrame): A pandas DataFrame with any number of columns.

        Returns:
        pd.DataFrame: A DataFrame with three columns - 'Prot1', 'Prot2', and 'matching_nonNaN_count'.
                      'Prot1' and 'Prot2' represent the column pairs, and 'matching_nonNaN_count'
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
        return pd.DataFrame(pairwise_non_nan_values, columns=['Prot1', 'Prot2', 'matching_nonNaN_count'])

    def derive_pvalues(self, correlations, n_samples):
        """
        Calculate p-values from correlations given the number of samples used to calculate the correlations.

        Parameters:
        - correlations: Correlation values.
        - n_samples: Number of samples.

        Returns:
        - pvals: Calculated p-values.
        """
        # Define constants for avoiding magic numbers
        PERFECT_POSITIVE_CORR_CORRECTION = 0.9999
        PERFECT_NEGATIVE_CORR_CORRECTION = -0.9999
        ZERO_CORR_CORRECTION = 0.0001

        pvals = []
        for correlation, n_sample in zip(correlations, n_samples):

            if np.isclose(correlation, 1):
                rf = PERFECT_POSITIVE_CORR_CORRECTION
            elif np.isclose(correlation, -1):
                rf = PERFECT_NEGATIVE_CORR_CORRECTION
            elif np.isclose(correlation, 0):
                rf = ZERO_CORR_CORRECTION
            else:
                rf = correlation

            df = n_sample - 2
            ts = rf * rf * (df / (1 - rf * rf))

            pval = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts)) if df > 2 else 1
            pvals.append(pval)

        return pvals

    def derive_pvalues_accuracies(self, accuracies, n_samples, df: pd.DataFrame):
        """
        Calculate p-values from accuracy scores for imbalanced datasets.

        Parameters:
        - accuracies: Accuracy values (as proportions, e.g., 0.9 for 90% accuracy).
        - n_samples: Number of samples used to calculate each accuracy score.
        - df: Dataframe to calculate the proportion of the majority class in the dataset.

        Returns:
        - pvals: Calculated p-values for each accuracy score, considering dataset imbalance.
        """
        pvals = []
        for accuracy, n_sample in zip(accuracies, n_samples):
            # Number of observed successes
            observed_successes = int(round(accuracy * n_sample))

            # Adjusted null hypothesis accuracy for imbalanced datasets
            null_accuracy = df.stack().value_counts(normalize=True).iloc[0]

            # Calculate the binomial test p-value
            pval = binomtest(observed_successes, n_sample, null_accuracy)
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
