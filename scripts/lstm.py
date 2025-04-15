import argparse
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


def build_refined_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.005)),
        Dropout(0.5),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM on preprocessed features")
    parser.add_argument("--data_dir", required=True, help="Path to processed dataset folder")
    parser.add_argument("--model_output", default="mobilenet_lstm_model.h5", help="Path to save trained model")
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_output

    # Load data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    class_names = np.load(os.path.join(data_dir, "class_names.npy"), allow_pickle=True)

    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    # Normalize
    X_train = X_train / np.max(X_train)
    X_val = X_val / np.max(X_val)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_refined_lstm_model(input_shape, num_classes)

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
    class_weights = dict(enumerate(class_weights))

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=[lr_scheduler, early_stopping],
        class_weight=class_weights
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=16)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, batch_size=16)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
