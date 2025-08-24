import os, yaml
from src.data_loader import load_config, load_data_npz, make_datasets
from pathlib import Path
from models.lstm_ntu import build_model
import tensorflow as tf

def main(cfg_path="configs/default.yaml"):
    cfg = load_config(cfg_path)
    (x_train, y_train), (x_test, y_test) = load_data_npz(cfg["data_path"])
    train_dataset, test_dataset = make_datasets(x_train, y_train, x_test, y_test, cfg["batch_size"])
    
    model = build_model(seq_len=cfg["seq_len"], feature_dim=cfg["feature_dim"], 
                        num_classes=cfg["num_classes"], lr=cfg["lr"])
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt = "checkpoints/lstm_ntu_best.h5"

    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
            
    ]

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=cfg["epochs"],
        callbacks=cb
    )
    model.save(ckpt.replace("_best", "_final"))

if __name__ == "__main__":
    main()