import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import models, transforms
from scipy.spatial import distance
import argparse
from PIL import Image
# Initialize Mediapipe and pre-trained models
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose()
face = mp_face.FaceMesh()

# L1 distance
def l1_distance(img1, img2):
    return np.mean(np.abs(img1 - img2))

# Keypoint extraction
def extract_keypoints(image, model, is_face=False):
    if is_face:
        results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    keypoints.append((landmark.x, landmark.y))
        return keypoints
    else:
        results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append((landmark.x, landmark.y, landmark.z))
        return keypoints

# Average Keypoint Distance (AKD)
def average_keypoint_distance(kp1, kp2):
    return np.mean([distance.euclidean(k1, k2) for k1, k2 in zip(kp1, kp2)])

# Missing Keypoint Rate (MKR)
def missing_keypoint_rate(kp1, kp2):
    kp1_set = set(kp1)
    kp2_set = set(kp2)
    return 1 - len(kp1_set - kp2_set) / len(kp1_set)

# Average Euclidean Distance (AED)
def average_euclidean_distance(id1, id2):
    return np.linalg.norm(id1 - id2)

# Load pre-trained re-identification models
face_reid_model = models.resnet18(pretrained=True)
body_reid_model = models.resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_identity(image, model):
    # Convert NumPy array to PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        identity = model(image)
    return identity

# Process videos
def process_videos(original_video_path, generated_video_path):
    original_video = cv2.VideoCapture(original_video_path)
    generated_video = cv2.VideoCapture(generated_video_path)
    
    metrics = {
        "L1": [],
        "AKD": [],
        "MKR": [],
        "AED": []
    }

    while original_video.isOpened() and generated_video.isOpened():
        ret1, frame1 = original_video.read()
        ret2, frame2 = generated_video.read()

        if not ret1 or not ret2:
            break

        # Resize frames for consistent comparison
        frame1 = cv2.resize(frame1, (256, 256))
        frame2 = cv2.resize(frame2, (256, 256))

        # L1 distance
        metrics["L1"].append(l1_distance(frame1, frame2))

        # Keypoint extraction and metrics
        kp1_pose = extract_keypoints(frame1, pose)
        kp2_pose = extract_keypoints(frame2, pose)
        kp1_face = extract_keypoints(frame1, face, is_face=True)
        kp2_face = extract_keypoints(frame2, face, is_face=True)

        if kp1_pose and kp2_pose:
            metrics["AKD"].append(average_keypoint_distance(kp1_pose, kp2_pose))
            metrics["MKR"].append(missing_keypoint_rate(kp1_pose, kp2_pose))
        if kp1_face and kp2_face:
            metrics["AKD"].append(average_keypoint_distance(kp1_face, kp2_face))
            metrics["MKR"].append(missing_keypoint_rate(kp1_face, kp2_face))

        # Identity extraction and AED
        id1_body = extract_identity(frame1, body_reid_model)
        id2_body = extract_identity(frame2, body_reid_model)
        id1_face = extract_identity(frame1, face_reid_model)
        id2_face = extract_identity(frame2, face_reid_model)

        metrics["AED"].append(average_euclidean_distance(id1_body, id2_body))
        metrics["AED"].append(average_euclidean_distance(id1_face, id2_face))

    # Average metrics over all frames
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    original_video.release()
    generated_video.release()
    return metrics

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video reconstruction quality.")
    parser.add_argument("original_video_path", type=str, help="Path to the original driving video.")
    parser.add_argument("generated_video_path", type=str, help="Path to the generated video.")
    
    args = parser.parse_args()
    
    metrics = process_videos(args.original_video_path, args.generated_video_path)
    print(metrics)

