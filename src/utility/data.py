import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import defaultdict

def make_date_dict(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    dates = df[['Year', 'Week']].drop_duplicates().to_numpy()
    dates = dates[dates[:, 1].argsort()]
    dates = dates[dates[:, 0].argsort(kind='stable')]
    date_dict = defaultdict()
    for i, week in enumerate(dates):
        date_dict[tuple(week)] = i
    return dates, date_dict

def make_type_dict(types: List[str]) -> dict:
    type_dict = defaultdict()
    for i, type_ in enumerate(types):
        type_dict[type_] = i
    return type_dict