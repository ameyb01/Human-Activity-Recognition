import os
import argparse
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_features_mobilenet(input_folder, output_folder, image_size=(224, 224)):
    """
    Extracts features from frames using MobileNet.

    Args:
        input_folder (str): Path to the folder containing frames organized by category.
        output_folder (str): Path to save extracted features.
        image_size (tuple): Target size for input images (default is (224, 224)).
    """
    model = MobileNet(weights="imagenet", include_top=False)

    for split in ["train", "val"]:
        split_path = os.path.join(input_folder, split)
        split_output_path = os.path.join(output_folder, split)

        categories = os.listdir(split_path)
        for category in categories:
            category_path = os.path.join(split_path, category)
            category_output_path = os.path.join(split_output_path, category)
            os.makedirs(category_output_path, exist_ok=True)

            frames = [f for f in os.listdir(category_path) if f.endswith(".jpg")]
            for frame_file in frames:
                frame_path = os.path.join(category_path, frame_file)
                img = load_img(frame_path, target_size=image_size)
                img_array = img_to_array(img)
                img_array = preprocess_mobilenet(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                features = model.predict(img_array)
                feature_file = os.path.join(category_output_path, f"{os.path.splitext(frame_file)[0]}.npy")
                np.save(feature_file, features)
                print(f"Extracted MobileNet features for {frame_file} in {category}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MobileNet features from image frames")
    parser.add_argument("--input_dir", required=True, help="Path to input frames folder (contains train/val folders)")
    parser.add_argument("--output_dir", required=True, help="Path to save extracted feature .npy files")
    args = parser.parse_args()

    extract_features_mobilenet(args.input_dir, args.output_dir)
