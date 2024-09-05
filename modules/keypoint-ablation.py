import cv2
import torch
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import sys
sys.path.append('modules')
from keypoint_detector import KPDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Process a video and detect keypoints.")
    parser.add_argument('--video_path', type=str, help='Optional path to the input video file.', default=None)

    parser.add_argument('--image_path', type=str, help='Optional path to input image file.', default=None)
    return parser.parse_args()

def process_image(image_path, output_image_path, kp_detecor):
    #Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    #Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Transform the image for the model input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply the transformation and add a batch dimension
    input_tensor = transform(rgb_image)
    input_tensor = input_tensor.unsqueeze(0) #Add batch dimensions

    # Detect keypoints
    with torch.no_grad():
        keypoints = kp_detector(input_tensor)['fg_kp'][0].cpu().numpy()

        
        # Convert keypoints from [-1, 1] to frame dimensions
    frame_width, frame_height = image.shape[1], image.shape[0]
    keypoints[:, 0] = (keypoints[:, 0] + 1) / 2 * frame_width
    keypoints[:, 1] = (keypoints[:, 1] + 1) / 2 * frame_height

        #Draw keypoints on frame
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    cv2.imwrite(output_image_path, image)
def process_video(video_path, output_path, kp_detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
   
    #Get frame dimensions for video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or error reading the frame.")
            break

        #Convert BGR to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform the frame for the model input
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  # Ensure this closing parenthesis is here

        # Apply the transformation and add a batch dimension
        input_tensor = transform(rgb_frame).unsqueeze(0)  # Correct comment alignment


        # Detect keypoints
        with torch.no_grad():
            keypoints = kp_detector(input_tensor)['fg_kp'][0].cpu().numpy()

        
        # Convert keypoints from [-1, 1] to frame dimensions
        keypoints[:, 0] = (keypoints[:, 0] + 1) / 2 * frame_width
        keypoints[:, 1] = (keypoints[:, 1] + 1) / 2 * frame_height

        #Draw keypoints on frame
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        #Write frame with keypoints to the output video
        out.write(frame)
        
        #Show in the frame
       #  cv2.imshow('Keypoints', frame)
       #  if cv2.waitKey(25) & 0xFF == ord('q'):
       #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete and saved to", os.path.abspath(output_path))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    kp_detector = KPDetector(num_tps=10)
    kp_detector.eval()
    if args.video_path:
        output_path = 'output_keypoints.mp4'
        process_video(args.video_path, output_path, kp_detector)
    if args.image_path:
        output_image_path = 'output_image_keypoints.png'
        process_image(args.image_path, output_image_path, kp_detector)
