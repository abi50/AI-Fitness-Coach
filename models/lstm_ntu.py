import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(seq_len=300, feature_dim=150, num_classes=60, lr=1e-3):
    model= models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(seq_len, feature_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model