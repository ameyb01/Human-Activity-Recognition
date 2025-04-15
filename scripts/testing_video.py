import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobilenet

# Define activity labels
activity_labels = {
    0: "Hit",
    1: "Jump",
    2: "Walk",
    3: "Wave",
}

# Mapping model types to extractors and preprocessors
FEATURE_EXTRACTORS = {
    "resnet": (ResNet50, preprocess_resnet, 7 * 7 * 2048),
    "vgg": (VGG16, preprocess_vgg, 7 * 7 * 512),
    "mobilenet": (MobileNet, preprocess_mobilenet, 7 * 7 * 1024),
}


def preprocess_frame(frame, preprocess_fn):
    frame_resized = cv2.resize(frame, (224, 224))
    return preprocess_fn(np.expand_dims(frame_resized, axis=0))


def predict_on_video(video_path, sequence_length, model, feature_extractor, preprocess_fn, activity_labels):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed = preprocess_frame(frame, preprocess_fn)
        features = feature_extractor.predict(preprocessed).flatten()
        frame_buffer.append(features)

        if len(frame_buffer) == sequence_length:
            input_seq = np.expand_dims(frame_buffer, axis=0)
            pred = model.predict(input_seq)
            predicted_class = np.argmax(pred, axis=1)[0]
            predictions.append(predicted_class)

            activity_name = activity_labels.get(predicted_class, "Unknown")
            cv2.putText(frame, f"Activity: {activity_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_buffer.pop(0)

        cv2.imshow("Activity Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Predicted Classes: {predictions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HAR on a video using pre-trained Transformer model")
    parser.add_argument("--model_type", type=str, choices=["resnet", "vgg", "mobilenet"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=30)

    args = parser.parse_args()

    # Load feature extractor and model
    extractor_class, preprocess_fn, feature_size = FEATURE_EXTRACTORS[args.model_type]
    feature_extractor = extractor_class(weights="imagenet", include_top=False)
    model = load_model(args.model_path)

    # Run prediction
    predict_on_video(args.video_path, args.sequence_length, model, feature_extractor, preprocess_fn, activity_labels)
