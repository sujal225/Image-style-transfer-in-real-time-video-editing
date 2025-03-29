import cv2
import os
import tkinter as tk
from tkinter import filedialog
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
    return count

def apply_style_transfer(style_transfer_script, model_path, input_folder, output_folder):
    """Apply style transfer using style_transfer.py for each frame."""
    os.makedirs(output_folder, exist_ok=True)

    for frame_name in sorted(os.listdir(input_folder)):
        if not frame_name.endswith(".jpg"):
            continue

        input_path = os.path.join(input_folder, frame_name)
        output_path = os.path.join(output_folder, frame_name)

        subprocess.run([
            "python", style_transfer_script, "eval",
            "--content-image", input_path,
            "--output-image", output_path,
            "--model", model_path,
            "--cuda", "0"  # Change to "1" if using GPU
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

# GUI to Select Files
def select_files():
    global video_file, output_folder, model_file

    # Open file picker for video
    video_file = filedialog.askopenfilename(title="Select Input Video", filetypes=[("MP4 files", "*.mp4"), ("All Files", "*.*")])
    if not video_file:
        return

    # Open directory picker for output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        return

    # Open file picker for model file (.pth)
    model_file = filedialog.askopenfilename(title="Select Style Transfer Model", filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
    if not model_file:
        return

    # Run the processing pipeline
    process_video()

# Process Video after Selecting Files
def process_video():
    frames_input = os.path.join(output_folder, "frames_original")
    frames_output = os.path.join(output_folder, "frames_styled")
    output_video = os.path.join(output_folder, "styled_output.mp4")

    num_frames = extract_frames(video_file, frames_input)
    apply_style_transfer("style_transfer.py", model_file, frames_input, frames_output)
    create_video(frames_output, output_video, fps=30)

    print("Process completed!")

# GUI Window
root = tk.Tk()
root.title("Style Transfer Video Processor")
root.geometry("400x300")

# Create a Select Button
select_button = tk.Button(root, text="Select Files & Process Video", command=select_files, height=3, width=30)
select_button.pack(pady=50)

root.mainloop()
