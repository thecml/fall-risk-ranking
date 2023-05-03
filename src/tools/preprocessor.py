import pandas as pd
from typing import List

def series_to_moving_average(df_ts: pd.DataFrame,
                             window_len: int,
                             lag: int,
                             lbl_cols: List) -> pd.DataFrame:
    cols = list(df_ts.columns.drop(lbl_cols))
    total_df = pd.DataFrame()
    for period in df_ts['Period'].unique():
        period_df = df_ts.loc[df_ts['Period'] == period].copy(deep=True) # Get period data
        for ft_col in cols:
            roll = period_df.groupby(['Id', 'Period'])[ft_col].rolling(window_len)
            ma = roll.mean().shift(lag).reset_index(0, drop=True)[period]
            period_df[ft_col] = ma
        period_df = period_df.dropna().reset_index(drop=True)
        total_df = pd.concat([total_df, period_df], axis=0)
    return total_df