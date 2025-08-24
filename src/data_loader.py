import numpy as np
import tensorflow as tf
import yaml, os


def load_config(path="configs/default.yaml"):
    # להדפיס נתיב מוחלט לעזרה בדיבוג
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config not found at: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        text = f.read()
    cfg = yaml.safe_load(text) or {}
    return cfg

def load_data_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    x_train, y_train = d['x_train'].astype("float32"), d['y_train'].astype(np.float32)
    x_test, y_test = d['x_test'].astype("float32"), d['y_test'].astype(np.float32)
    max_val = np.max(np.abs(x_train))
    x_train /= (max_val if max_val != 0 else 1.0)
    x_test  /= (max_val if max_val != 0 else 1.0)
    return (x_train, y_train), (x_test, y_test)

def make_datasets(x_train, y_train, x_test, y_test, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(min(2000, len(x_train))).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, test_dataset