import pandas as pd
import tbutils.seed
from tqdm import tqdm


# Aggregation functions
def sum_df(df: pd.DataFrame) -> pd.Series:
    return df.sum(axis=1, skipna=True)


def mean_df(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1, skipna=True)


def median_df(df: pd.DataFrame) -> pd.Series:
    return df.median(axis=1, skipna=True)


def max_df(df: pd.DataFrame) -> pd.Series:
    return df.max(axis=1, skipna=True)


def min_df(df: pd.DataFrame) -> pd.Series:
    return df.min(axis=1, skipna=True)


def std_df(df: pd.DataFrame) -> pd.Series:
    return df.std(axis=1, skipna=True, ddof=0)


def sem_df(df: pd.DataFrame) -> pd.Series:
    return df.sem(axis=1, skipna=True, ddof=0)


def range_df(df: pd.DataFrame) -> pd.Series:
    return df.max(axis=1, skipna=True) - df.min(axis=1, skipna=True)


def diff_df(df: pd.DataFrame) -> pd.Series:
    # First column minus 2nd column
    assert df.shape[1] == 2, "Diff operation only works for 2 columns."
    return df.iloc[:, 0] - df.iloc[:, 1]


dict_agg_functions = {
    "sum": sum_df,
    "mean": mean_df,
    "median": median_df,
    "max": max_df,
    "min": min_df,
    "std": std_df,
    "sem": sem_df,
    "range": range_df,
    # Only works for certain shapes
    "diff": diff_df,
    "abs_diff": lambda df: diff_df(df).abs(),
}


# Deltas functions
def std_deltas_df(df: pd.DataFrame, mean_values: pd.Series) -> pd.Series:
    std_dataframe = df.std(axis=1, skipna=True, ddof=0)
    return std_dataframe, std_dataframe


def sem_deltas_df(df: pd.DataFrame, mean_values: pd.Series) -> pd.Series:
    sem_dataframe = df.sem(axis=1, skipna=True, ddof=0)
    return sem_dataframe, sem_dataframe


def range_deltas_df(df: pd.DataFrame, mean_values: pd.Series) -> pd.Series:
    return (
        mean_values - df.min(axis=1, skipna=True),
        df.max(axis=1, skipna=True) - mean_values,
    )


def iqr_deltas_df(df: pd.DataFrame, mean_values: pd.Series) -> pd.Series:
    return (
        mean_values - df.quantile(0.25, axis=1, interpolation="linear"),
        df.quantile(0.75, axis=1, interpolation="linear") - mean_values,
    )


def none_deltas_df(df: pd.DataFrame, mean_values: pd.Series) -> pd.Series:
    return None, None


dict_deltas_functions = {
    "std": std_deltas_df,
    "sem": sem_deltas_df,
    "range": range_deltas_df,
    "iqr": iqr_deltas_df,
    "none": none_deltas_df,
}
