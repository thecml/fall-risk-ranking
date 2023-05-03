from pathlib import Path
from typing import Any
import pickle
import pandas as pd
import tensorflow as tf
import joblib

def write_joblib(path: Path, data):
    joblib.dump(data, path)

def write_model(path: Path, model: tf.keras.Model):
    model.save(path)

def write_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)

def write_pickle(path: Path, obj: Any):
    """
    Pickles the given object at the given location

    :param path: path to where the object should be pickled
    :param obj: the object to be pickled
    """
    file_handler = open(path, 'wb')
    pickle.dump(obj, file_handler)
    file_handler.close()
    print(f"-- Pickled object at  {path} --")
