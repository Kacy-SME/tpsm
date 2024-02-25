import cv2

# Specify the path to your video file
video_path = r'C:\Users\kacy\Documents\TPSM\tpsm\checkpoints\Processed-Wasp\SR5.mp4'  # Replace with the actual path to your video

# Initialize a frame count variable
frame_count = 0

# Specify the desired frame shape (width, height)
width = 256  # Replace with your desired width
height = 256  # Replace with your desired height
frame_shape = (width, height)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
else:
    # Loop through the frames and count them
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop when no more frames are available
        
        # Check if frame_shape dimensions are valid
        if width > 0 and height > 0:
            # Ensure frame dimensions are valid
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                # Resize the frame
                resized_frame = cv2.resize(frame, frame_shape)
                # Rest of your processing here
        else:
            print("Invalid frame dimensions provided.")

        frame_count += 1

    # Close the video file
    cap.release()

    # Print the total number of frames
    print("Total number of frames:", frame_count)
