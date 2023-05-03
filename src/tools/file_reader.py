import pickle
from typing import Any, List, Tuple
from pathlib import Path
import config as cfg
import pandas as pd
import tensorflow as tf
import joblib

def read_joblib(path: Path):
    return joblib.load(path)

def read_model(path: Path):
    return tf.keras.models.load_model(path)

def read_csv(path: Path,header: str='infer',
             sep: str=',', usecols: List[int]=None,
             names: List[str]=None, converters: dict=None,
             encoding=None, skiprows=None, parse_dates=None) -> pd.DataFrame:

    return pd.read_csv(path, header=header, sep=sep, usecols=usecols,
                       names=names, converters=converters,
                       encoding=encoding, skiprows=skiprows,
                       parse_dates=parse_dates)

def read_pickle(path: Path) -> Any:
    """
    Loads the pickled object at the location given.

    :param path: Path (including the file itself)
    :return: obj
    """
    file_handler = open(path, 'rb')
    obj = pickle.load(file_handler)
    file_handler.close()
    return obj