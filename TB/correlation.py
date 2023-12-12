import scipy.stats
from statsmodels.stats import multitest
import pandas as pd


class Diff_Network_Analysis:

    def __init__(self, stateA: pd.DataFrame, stateB: pd.DataFrame, correlation: str = 'pearson',
                 significance_base_level: float = 0.01):
        if correlation not in ['pearson', 'spearman']:
            raise ValueError(f"{correlation} should be a 'pearson' or 'spearman.")

        self.stateA, self.stateB = stateA, stateB
        self.correlation = correlation
        self.significance_base_level = significance_base_level

    def diff_corr_network(self):

        df_r_stateA, df_r_stateB = self.corr_network(self.stateA), self.corr_network(self.stateB)

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
        df_r['p-value_corrected'] = pvalues_corrected

        df_r['r-value'] = df_r.apply(lambda x: np.nan if x['Prot1'] == x['Prot2'] else x['r-value'], axis=1)

        # df_r_mssa = (

        #     df_r_mssa
        #     [(df_r_mssa['p-value_corrected'] <= 0.01)]
        #     .dropna(subset='r-value')
        #     .sort_values(by='r-value')
        #     .reset_index(drop=True)
        # )

        # df_r_mssa = (

        #     df_r_mssa
        #     .set_index(['Prot1', 'Prot2'])
        #     .reindex(df_r_mssa.set_index(['Prot1', 'Prot2']).index)
        #     .reset_index()
        # )

        return df_r

    def derive_pvalues(self, correlations, n_samples):
        """
        Calculate pvalues from correlations given the number of samples used to calculate the correlations.
        Source: https://stackoverflow.com/a/24547964/991496
        """
        correlations = np.asarray(correlations)
        rf = correlations
        df = n_samples - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pvals = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
        return pvals

    def correct_pvalues(self, pvals):
        return multitest.fdrcorrection(pvals, alpha=self.significance_base_level, method='indep', is_sorted=False)
