import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np

# Initialize lists to store the clicked keypoints for both image and video
clicked_keypoints_image = []
clicked_keypoints_video = []

def on_click_image(event):
    """Callback function to capture clicked points on the source image."""
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        clicked_keypoints_image.append([x, y])
        print(f"Image: Clicked at: ({x}, {y})")
        plt.plot(x, y, 'ro')
        plt.draw()

def on_click_video(event, frame_idx):
    """Callback function to capture clicked points on the driving video frame."""
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        clicked_keypoints_video.append([frame_idx, x, y])
        print(f"Video frame {frame_idx}: Clicked at: ({x}, {y})")
        plt.plot(x, y, 'bo')
        plt.draw()

# Function to capture keypoints from source image
def process_image(image_path):
    global clicked_keypoints_image
    clicked_keypoints_image = []

    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image and capture clicks
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    fig.canvas.mpl_connect('button_press_event', on_click_image)
    plt.title("Click on the source image to define keypoints (close when done)")
    plt.show()

# Function to capture keypoints from driving video
def process_video(video_path):
    global clicked_keypoints_video
    clicked_keypoints_video = []

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for displaying with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame and capture clicks
        fig, ax = plt.subplots()
        ax.imshow(frame_rgb)
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click_video(event, frame_idx))
        plt.title(f"Click on frame {frame_idx} of the driving video (close when done)")
        plt.show()

        frame_idx += 1
        if frame_idx == 5:  # Only show a few frames for demonstration (adjust as needed)
            break

    cap.release()

# Function to save keypoints to an HDF5 file
def save_keypoints_to_h5(output_h5_file, keypoints_image, keypoints_video):
    with h5py.File(output_h5_file, 'w') as h5_file:
        h5_file.create_dataset('keypoints_image', data=keypoints_image)
        h5_file.create_dataset('keypoints_video', data=keypoints_video)
    print(f"Keypoints saved to {output_h5_file}")

if __name__ == "__main__":
    # Path to source image and driving video
    image_path = "../assets/interaction-source.png"
    video_path = "../assets/interaction-driving.mp4"
    output_h5_file = "clicked_keypoints.h5"

    # Process source image and driving video
    process_image(image_path)
    process_video(video_path)

    # Save the collected keypoints to an HDF5 file
    save_keypoints_to_h5(output_h5_file, clicked_keypoints_image, clicked_keypoints_video)

