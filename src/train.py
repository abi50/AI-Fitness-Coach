import os
import src.data_loader as data_loader
from pathlib import Path
from models.lstm_ntu import build_model
import tensorflow as tf
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--dry_run", action="store_true",
                   help="Build model + one forward pass on a single batch, then exit")
    return p.parse_args()

def get_callbacks(ckpt):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

def main(cfg_path="configs/default.yaml", dry_run=False):
    cfg = data_loader.load_config(cfg_path)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data_npz(cfg["data_path"])
    train_dataset, test_dataset = data_loader.make_datasets(x_train, y_train, x_test, y_test, cfg["batch_size"])
    
    model = build_model(seq_len=cfg["seq_len"], feature_dim=cfg["feature_dim"], 
                        num_classes=cfg["num_classes"], lr=cfg["lr"])
    Path(os.path.abspath("checkpoints")).mkdir(exist_ok=True)
    ckpt = cfg.get("checkpoint_path", "checkpoints/lstm_ntu_best.h5")

    cb = get_callbacks(ckpt)
    # --- DRY-RUN: Forward pass on one batch and exit ---
    if dry_run:
        # Take one batch from the train_dataset and check forward pass
        if not train_dataset:
            print("[DRY-RUN] The training dataset is empty. Exiting.")
            return
        xb, yb = next(iter(train_dataset))
        _ = model(xb, training=False)
        print("[DRY-RUN] One batch passed successfully.")
            
    

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=cfg["epochs"],
        callbacks=cb
    )

if __name__ == "__main__":
    args = parse_args()
    main(cfg_path=args.cfg_path, dry_run=args.dry_run)