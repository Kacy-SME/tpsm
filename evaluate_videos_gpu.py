import os
import cv2
import numpy as np
import argparse
import mediapipe as mp
import torch
from torchvision import models, transforms
from PIL import Image

# Set which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change the number to the desired GPU

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained re-identification models (ResNet in this case)
face_reid_model = models.resnet18(pretrained=True).cuda()
body_reid_model = models.resnet18(pretrained=True).cuda()

# Transformation for re-identification models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_keypoints(image, model, is_face=False):
    results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = []
    if is_face:
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    keypoints.append((landmark.x, landmark.y))
    else:
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append((landmark.x, landmark.y, landmark.z))
    return keypoints, results

def overlay_keypoints(image, results, is_face=False):
    if is_face:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))
    else:
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
    return image

def extract_identity(image, model):
    # Convert NumPy array to PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        identity = model(image)
    return identity.cpu().numpy()

def l1_distance(img1, img2):
    assert img1.shape == img2.shape, "Input images must have the same dimensions"
    return np.mean(np.abs(img1 - img2))

def calculate_akd(kp1, kp2):
    assert len(kp1) == len(kp2), "Keypoints lists must have the same length"
    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(kp1, kp2)]
    return np.mean(distances)

def calculate_mkr(kp1, kp2):
    missing_count = sum(1 for p1 in kp1 if p1 not in kp2)
    return missing_count / len(kp1)

def calculate_aed(id1, id2):
    return np.linalg.norm(id1 - id2)

def process_videos(original_video_path, generated_video_path, output_dir):
    original_video = cv2.VideoCapture(original_video_path)
    generated_video = cv2.VideoCapture(generated_video_path)
    
    l1_distances = []
    akd_distances = []
    mkr_rates = []
    aed_distances = []
    frame_count = 0

    pose = mp_pose.Pose()

    while original_video.isOpened() and generated_video.isOpened():
        ret1, frame1 = original_video.read()
        ret2, frame2 = generated_video.read()

        if not ret1 or not ret2:
            break

        # Resize frames to the same size
        frame1 = cv2.resize(frame1, (256, 256))
        frame2 = cv2.resize(frame2, (256, 256))

        # Extract keypoints
        kp1_pose, results1 = extract_keypoints(frame1, pose)
        kp2_pose, results2 = extract_keypoints(frame2, pose)

        # Overlay keypoints on the frames
        frame1_with_kp = overlay_keypoints(frame1.copy(), results1)
        frame2_with_kp = overlay_keypoints(frame2.copy(), results2)

        # Save frames with keypoints as PNG
        cv2.imwrite(f"{output_dir}/original_frame_{frame_count}.png", frame1_with_kp)
        cv2.imwrite(f"{output_dir}/generated_frame_{frame_count}.png", frame2_with_kp)

        # Calculate L1 distance
        l1_distances.append(l1_distance(frame1, frame2))

        # Calculate AKD
        if len(kp1_pose) == len(kp2_pose):
            akd_distances.append(calculate_akd(kp1_pose, kp2_pose))

        # Calculate MKR
        mkr_rates.append(calculate_mkr(kp1_pose, kp2_pose))

        # Extract identities and calculate AED
        id1 = extract_identity(frame1, body_reid_model)
        id2 = extract_identity(frame2, body_reid_model)
        aed_distances.append(calculate_aed(id1, id2))

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Average distances and rates over all frames
    avg_l1_distance = np.mean(l1_distances)
    avg_akd_distance = np.mean(akd_distances)
    avg_mkr_rate = np.mean(mkr_rates)
    avg_aed_distance = np.mean(aed_distances)

    original_video.release()
    generated_video.release()
    pose.close()
    print(f"Total frames processed: {frame_count}")
    return avg_l1_distance, avg_akd_distance, avg_mkr_rate, avg_aed_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video reconstruction quality.")
    parser.add_argument("original_video_path", type=str, help="Path to the original driving video.")
    parser.add_argument("generated_video_path", type=str, help="Path to the generated video.")
    parser.add_argument("output_dir", type=str, help="Directory to save output images.")
    
    args = parser.parse_args()
    
    avg_l1_distance, avg_akd_distance, avg_mkr_rate, avg_aed_distance = process_videos(args.original_video_path, args.generated_video_path, args.output_dir)
    print(f"Average L1 distance: {avg_l1_distance}")
    print(f"Average AKD distance: {avg_akd_distance}")
    print(f"Average MKR rate: {avg_mkr_rate}")
    print(f"Average AED distance: {avg_aed_distance}")

