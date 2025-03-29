import cv2
import os
import argparse
import subprocess

def extract_frames(video_path, output_folder):
    """Extract frames from a video and save them as images."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    
    cap.release()
    print(f"Extracted {count} frames from {video_path}.")
    return count  # Return number of frames

def apply_style_transfer(style_transfer_script, model_path, input_folder, output_folder):
    """Apply style transfer using style_transfer.py for each frame."""
    os.makedirs(output_folder, exist_ok=True)

    for frame_name in sorted(os.listdir(input_folder)):
        if not frame_name.endswith(".jpg"):
            continue

        input_path = os.path.join(input_folder, frame_name)
        output_path = os.path.join(output_folder, frame_name)

        # Run style_transfer.py on each frame
        subprocess.run([
            "python", style_transfer_script, "eval",
            "--content-image", input_path,
            "--output-image", output_path,
            "--model", model_path,
            "--cuda", "0"  # Change to 1 if using GPU
        ])
        print(f"Styled: {frame_name}")

def create_video(frame_folder, output_video, fps):
    """Recombine processed frames into a video."""
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        out.write(frame)

    out.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply style transfer to a video")
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, required=True, help="Output video file path")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--style-transfer", type=str, default="style_transfer.py", help="Path to style_transfer.py script")

    args = parser.parse_args()

    frames_input = "frames_original"
    frames_output = "frames_styled"

    num_frames = extract_frames(args.video, frames_input)
    apply_style_transfer(args.style_transfer, args.model, frames_input, frames_output)
    create_video(frames_output, args.output, args.fps)
