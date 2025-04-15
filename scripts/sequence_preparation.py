import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def group_frames_by_video(input_folder, sequence_length, stride=10):
    frame_files = sorted(os.listdir(input_folder))
    video_groups = {}

    for frame_file in frame_files:
        video_name = frame_file.split("_frame")[0]
        if video_name not in video_groups:
            video_groups[video_name] = []
        frame_path = os.path.join(input_folder, frame_file)
        frame_features = np.load(frame_path)
        frame_features = frame_features.reshape(-1)
        video_groups[video_name].append(frame_features)

    sequences = []
    video_ids = []
    for video_name, frames in video_groups.items():
        frames = np.array(frames)

        if len(frames) >= sequence_length:
            for start in range(0, len(frames) - sequence_length + 1, stride):
                sequences.append(frames[start:start + sequence_length])
                video_ids.append(video_name)
        else:
            padded_frames = pad_sequences([frames], maxlen=sequence_length, dtype="float32")[0]
            sequences.append(padded_frames)
            video_ids.append(video_name)

    return np.array(sequences), video_ids

def process_all_categories(input_folder, sequence_length, stride=10):
    X = []
    y = []
    class_names = sorted(os.listdir(input_folder))

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(input_folder, class_name)
        sequences, _ = group_frames_by_video(class_folder, sequence_length, stride)
        X.extend(sequences)
        y.extend([label] * len(sequences))

    return np.array(X), np.array(y), class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sequences from extracted frame features")
    parser.add_argument("--train_dir", required=True, help="Path to training feature directory")
    parser.add_argument("--val_dir", required=True, help="Path to validation feature directory")
    parser.add_argument("--output_dir", required=True, help="Path to save processed .npy files")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames per sequence")
    parser.add_argument("--stride", type=int, default=10, help="Sliding window stride")
    args = parser.parse_args()

    # Process training and validation data
    X_train, y_train, class_names = process_all_categories(args.train_dir, args.sequence_length, args.stride)
    X_val, y_val, _ = process_all_categories(args.val_dir, args.sequence_length, args.stride)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.output_dir, "class_names.npy"), class_names)

    print(f"Processed data saved in '{args.output_dir}'")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
