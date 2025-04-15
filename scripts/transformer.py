import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_refined_transformer(sequence_length, feature_size, num_classes):
    inputs = layers.Input(shape=(sequence_length, feature_size), name="input_layer")

    positional_encoding = layers.Dense(128, activation="relu", name="positional_encoding")(inputs)
    spatial_encoding = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu", name="spatial_encoding")(positional_encoding)
    x = layers.Add(name="add_positional_spatial")([positional_encoding, spatial_encoding])
    x = layers.LayerNormalization(name="layer_norm_encoding")(x)

    for i in range(3):
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1, name=f"multi_head_attention_{i}")(x, x)
        attention_output = layers.Add(name=f"residual_connection_attention_{i}")([x, attention_output])
        attention_output = layers.LayerNormalization(name=f"layer_norm_attention_{i}")(attention_output)

        ffn_output = layers.Dense(256, activation="relu", name=f"dense_ffn_1_{i}")(attention_output)
        ffn_output = layers.Dropout(0.3, name=f"dropout_ffn_1_{i}")(ffn_output)
        ffn_output = layers.Dense(128, activation="relu", name=f"dense_ffn_2_{i}")(ffn_output)
        x = layers.Add(name=f"residual_connection_ffn_{i}")([attention_output, ffn_output])
        x = layers.LayerNormalization(name=f"layer_norm_ffn_{i}")(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
    x = layers.Dropout(0.4, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = models.Model(inputs, outputs, name="Refined_Transformer")
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for HAR")
    parser.add_argument("--data_dir", required=True, help="Path to processed dataset folder")
    parser.add_argument("--model_output", required=True, help="Path to save the trained model (.h5)")
    args = parser.parse_args()

    # Load data
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))

    sequence_length = X_train.shape[1]
    feature_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    model = build_refined_transformer(sequence_length, feature_size, num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[early_stopping, lr_scheduler]
    )

    model.save(args.model_output)
    print(f"Model saved to {args.model_output}")

    # Evaluate
    train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=16)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, batch_size=16)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
