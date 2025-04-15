import cv2
import os
import argparse

def extract_frames_from_videos(input_folder, output_folder, frame_rate=3):
    """
    Extracts frames from .avi videos and saves them in the specified output folder.

    Args:
        input_folder (str): Path to the folder containing videos organized by category.
        output_folder (str): Path to save extracted frames.
        frame_rate (int): Number of frames to extract per second.
    """
    for split in ["train", "val"]:
        split_path = os.path.join(input_folder, split)
        split_output_path = os.path.join(output_folder, split)

        categories = os.listdir(split_path)
        for category in categories:
            category_path = os.path.join(split_path, category)
            category_output_path = os.path.join(split_output_path, category)
            os.makedirs(category_output_path, exist_ok=True)

            videos = [f for f in os.listdir(category_path) if f.endswith(".avi")]
            for video_file in videos:
                video_path = os.path.join(category_path, video_file)
                cap = cv2.VideoCapture(video_path)

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_interval = max(1, fps // frame_rate)

                frame_count = 0
                extracted_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:
                        frame_filename = os.path.join(
                            category_output_path,
                            f"{os.path.splitext(video_file)[0]}_frame_{extracted_count:04d}.jpg"
                        )
                        cv2.imwrite(frame_filename, frame)
                        extracted_count += 1
                    frame_count += 1

                cap.release()
                print(f"Extracted {extracted_count} frames from {video_file} in {category}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input_dir", required=True, help="Root folder of dataset (containing train/val folders)")
    parser.add_argument("--output_dir", required=True, help="Root folder to save extracted frames")
    parser.add_argument("--frame_rate", type=int, default=5, help="Number of frames to extract per second")

    args = parser.parse_args()

    extract_frames_from_videos(args.input_dir, args.output_dir, frame_rate=args.frame_rate)
