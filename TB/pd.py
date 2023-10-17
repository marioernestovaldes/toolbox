import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import copy
import matplotlib.pyplot as plt
import logging

def scale_dataframe(df, scaler="standard", **kwargs):
    """
    Scale all columns in a dense dataframe.
    :param df: Dataframe to scale
    :type df: pandas.DataFrame
    :param scaler: Scaler to use ['robust', 'standard'], defaults to "standard"
    :type scaler: str, optional
    :return: Scaled dataframe
    :rtype: pandas.DataFrame
    """
    df = df.copy()
    if scaler == "standard":
        scaler = StandardScaler(**kwargs)
    elif scaler == "robust":
        scaler = RobustScaler(**kwargs)
    df.loc[:, :] = scaler.fit_transform(df)
    return df

def col_to_class(df, col_name, possible_values=None, delete_col=True):
    """
    Convert a categorical column into binary class columns.

    Parameters:
    - df: DataFrame
    - col_name: Name of the column to be converted.
    - possible_values: List of possible values for the column, if None, it's extracted from the column.
    - delete_col: If True, delete the original column.

    Returns:
    - DataFrame with binary class columns.
    """
    print("Converting %s to classes" % col_name)
    tmp = df.loc[:, []].copy()
    if possible_values is None:
        possible_values = df[col_name].value_counts().index
    for value in possible_values:
        new_col_name = col_name + "_" + str(value)
        tmp.loc[:, new_col_name] = (df.loc[:, col_name] == value).astype(int)
    tmp = tmp[sorted(tmp.columns)]
    if delete_col:
        df.drop(col_name, 1, inplace=True)
    return df.join(tmp)

def reduce_mem_usage(df, verbose=True):
    """
    Reduces the memory usage of a DataFrame by down-casting numeric data types.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def get_duplicate_col_values(df, col_name):
    """
    Get duplicate rows in a DataFrame based on a specified column.
    """
    val_counts = df[col_name].value_counts()
    values = val_counts[val_counts > 1].index
    duplicates = df[df[col_name].isin(values)].sort_values(col_name)
    return duplicates

def sort_df_by_row_count(df, axis=1, ascending=True):
    """
    Sort a DataFrame's columns by the sum of their values in ascending or descending order.
    """
    ndx = df.sum(axis=axis).sort_values(ascending=ascending).index
    return df[ndx]

def stratify_df(df, columns, n_sample=None, random_state=None):
    """
    Stratify a DataFrame based on specified columns to ensure a balanced dataset.
    """
    count_per_group = df.groupby(columns).count().iloc[:, 0]
    if n_sample is None:
        n_sample = count_per_group.min().min()
        print(f"Using n_sample={n_sample}.")
    stratified = df.groupby(columns, group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_sample), random_state=random_state)
    )
    return stratified

def val_count_df(df, column_name, sort_by_column_name=False):
    """
    Generate a DataFrame that contains value counts and percentages for a specified column in the original DataFrame.
    """
    value_count = (
        df[column_name]
        .value_counts()
        .reset_index()
        .rename(columns={column_name: "Value Count", "index": column_name})
        .set_index(column_name)
    )
    value_count["Percentage"] = df[column_name].value_counts(normalize=True) * 100
    value_count = value_count.reset_index()
    if sort_by_column_name:
        value_count = value_count.sort_values(column_name)
    return value_count

def plot_and_display_valuecounts(df, column_name, sort_by_column_name):
    """
    Display a pie chart of value counts for a specified column in the DataFrame.
    """
    val_count = val_count_df(df, column_name, sort_by_column_name)
    display(val_count)
    val_count.set_index(column_name).plot.pie(
        y="Value Count", figsize=(5, 5), legend=False, ylabel=""
    )

def plot_and_display_compare_valuecounts(df1, df2, column_name, sort_by_column_name):
    """
    Compare and display value counts for a specified column between two DataFrames, showing the value counts as pie charts.
    """
    val_count_1 = val_count_df(df1, column_name, sort_by_column_name)
    val_count_1 = val_count_1.rename(
        columns={"Value Count": "train_value_count", "Percentage": "train_percentage"}
    )
    val_count_2 = val_count_df(df2, column_name, sort_by_column_name)
    val_count_2 = val_count_2.rename(
        columns={"Value Count": "test_value_count", "Percentage": "test_percentage"}
    )

    val_count = pd.merge(val_count_1, val_count_2, on=column_name, how="outer")
    val_count = val_count.fillna(
        0
    )  # if the data is missing from a column, there is none so we fill with 0's
    display(val_count)

    val_count = val_count.drop(
        columns=["train_percentage", "test_percentage"]
    )  # avoid duplicating pie plots
    val_count.set_index(column_name).plot.pie(
        figsize=(12, 7), legend=False, ylabel="", subplots=True, title=["Train", "Test"]
    )