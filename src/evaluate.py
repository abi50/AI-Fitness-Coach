from src.data_loader import load_config, load_data_npz, make_datasets
from tensorflow.keras.models import load_model

def main():
    cfg = load_config()
    (_, _), (x_test, y_test) = load_data_npz(cfg["data_path"])
    _, test_dataset = make_datasets(x_test, y_test, x_test, y_test, cfg["batch_size"])
    model = load_model("checkpoints/lstm_ntu_best.h5")
    loss, acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()