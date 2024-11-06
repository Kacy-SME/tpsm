import matplotlib
matplotlib.use('Agg')
import sys
import yaml
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network_demo import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion_demo import DenseMotionNetwork
from modules.avd_network import AVDNetwork
import subprocess
import argparse
import h5py
import torch
import logging
import pdb
from tqdm import tqdm

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")


# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def load_keypoints_from_hdf5(hdf5_file, frame_idx):
    with h5py.File(hdf5_file, 'r') as f:
        if 'keypoints' not in f:
            logging.error("The 'keypoints' dataset is missing in the HDF5 file.")
            return None
        
        keypoints = f['keypoints']
        
        # Check if the frame index is within bounds
        if frame_idx >= keypoints.shape[3]:
            logging.error(f"Requested frame index {frame_idx} exceeds available frames in 'keypoints'.")
            return None
        
        keypoints_for_frame = keypoints[:, :, :, frame_idx]
        logging.info(f"Loaded keypoints for frame {frame_idx}: shape {keypoints_for_frame.shape}")
        logging.debug(f"Keypoints data for frame {frame_idx}:\n{keypoints_for_frame}")

        # Print or log specific values for inspection
        if frame_idx < 5 or frame_idx in [100, 500, 1000]:  # Adjust these indices to sample more frames
            logging.info(f"Sample data from frame {frame_idx}: {keypoints_for_frame[:5]}")  # Log first 5 instances for brevity

    return keypoints_for_frame


def pad_kp_driving(kp_driving, batch_size):
    num_instances = kp_driving.shape[0]
    remainder = num_instances % batch_size
    if remainder != 0:
        padding_needed = batch_size - remainder
        pad_shape = (padding_needed,) + kp_driving.shape[1:]
        padding = np.zeros(pad_shape)
        kp_driving_padded = np.concatenate((kp_driving, padding), axis=0)
        return kp_driving_padded
    return kp_driving

def load_and_preprocess_keypoints(hdf5_file, batch_size, frame_idx=0):
    kp_driving = load_keypoints_from_hdf5(hdf5_file, frame_idx)
    if kp_driving is None:
        logging.error("Failed to load keypoints for the specified frame.")
        return None, None

    kp_driving = np.nan_to_num(kp_driving, nan=0.0)
    kp_driving_padded = pad_kp_driving(kp_driving, batch_size)
    logging.info(f"Padded kp_driving shape: {kp_driving_padded.shape}")

    kp_driving_tensor = torch.tensor(kp_driving_padded).float()
    kp_source_tensor = kp_driving_tensor.clone()

    # Align tensors for `kp_source` and `kp_driving`
    kp_driving_reshaped = kp_driving_tensor.view(batch_size, -1, 2)
    kp_source_reshaped = kp_source_tensor.view(batch_size, -1, 2)
    logging.info(f"Reshaped kp_driving: {kp_driving_reshaped.shape}")

    kp_source_dict = {'fg_kp': kp_source_reshaped}
    kp_driving_dict = {'fg_kp': kp_driving_reshaped}
    logging.info(f"Preprocessed keypoints: kp_source_dict {kp_source_dict}, kp_driving_dict {kp_driving_dict}")
    return kp_source_dict, kp_driving_dict

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    # Initialize networks
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    
    # Move models to device
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load weights with strict=False to handle missing keys
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    dense_motion_network.load_state_dict(checkpoint.get('dense_motion_network', {}), strict=False)

    # Initialize missing parameters for `conv_to_3` if needed
    if hasattr(dense_motion_network, 'conv_to_3'):
        if 'conv_to_3.weight' not in checkpoint['dense_motion_network']:
            nn.init.kaiming_normal_(dense_motion_network.conv_to_3.weight, mode='fan_out', nonlinearity='relu')
        if 'conv_to_3.bias' not in checkpoint['dense_motion_network']:
            nn.init.constant_(dense_motion_network.conv_to_3.bias, 0)

    # Load AVD network if available
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    # Set to eval mode
    inpainting.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, dense_motion_network, avd_network

def relative_kp(kp_source, kp_driving, kp_driving_initial):
    # Extract the tensors from dictionaries if necessary
    kp_driving_tensor = kp_driving['fg_kp'] if isinstance(kp_driving, dict) else kp_driving
    kp_driving_initial_tensor = kp_driving_initial['fg_kp'] if isinstance(kp_driving_initial, dict) else kp_driving_initial

    # Calculate areas and movement scale
    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial_tensor[0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    # Compute difference and apply scaling
    kp_value_diff = (kp_driving_tensor - kp_driving_initial_tensor) * adapt_movement_scale
    kp_new_fg_kp = kp_value_diff + kp_source['fg_kp']

    # Wrap result in a dictionary to match expected structure
    kp_new = {'fg_kp': kp_new_fg_kp}
    return kp_new

# Define make_animation function here
def make_animation(source_image, driving_video, inpainting_network, dense_motion_network, avd_network, device, kp_source, kp_driving_initial, mode='relative'):
    # Log key information only to reduce log size
    logging.info(f"kp_driving_initial type: {type(kp_driving_initial)}, shape: {kp_driving_initial['fg_kp'].shape if 'fg_kp' in kp_driving_initial else 'N/A'}")

    logging.info("Starting animation...")

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)

        for frame_idx in tqdm(range(len(driving_video)), desc="Animating Frames"):
            driving_frame = driving_video[frame_idx].to(device)
            kp_driving = kp_driving_initial['fg_kp'][frame_idx] if frame_idx < kp_driving_initial['fg_kp'].shape[0] else None
            if kp_driving is None:
                logging.warning(f"Skipping frame {frame_idx}: kp_driving is None.")
                continue
            
            kp_norm = kp_driving if mode == 'standard' else relative_kp(kp_source, kp_driving, kp_driving_initial)
            # Adjust channels if necessary
            if source.shape[1] != 84:
                source = nn.Conv2d(source.shape[1], 84, kernel_size=1).to(device)(source)

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm, kp_source=kp_source, bg_param=None, dropout_flag=False)
            logging.info(f"dense_motion keys: {list(dense_motion.keys())}")
            out = inpainting_network(source, dense_motion)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    logging.info("Animation complete.")
    logging.info(f"make_animation returns predictions with {len(predictions)} frames.")
    return predictions

def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--addaudio', action=argparse.BooleanOptionalAction)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='./assets/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image that the model was trained on.')
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'],
                        help="Animation mode: ['standard', 'relative', 'avd']")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Align to the most similar frame in the driving video (only for faces).")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="Run on CPU.")
    parser.add_argument("--keypoint_hdf5", required=True, help="Path to the HDF5 file with precomputed keypoints")

    opt = parser.parse_args()
    
    # Load and preprocess keypoints
    batch_size = 16  # Define or load batch size from your config
    kp_source, kp_driving_initial = load_and_preprocess_keypoints(opt.keypoint_hdf5, batch_size)

    # Load the source and driving videos
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    # Set device
    device = torch.device('cpu' if opt.cpu else 'cuda')
    
    # Resize images
    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = torch.tensor([resize(frame, opt.img_shape)[..., :3] for frame in driving_video]).float()
 
    # Load pre-trained models
    inpainting, dense_motion_network, avd_network = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, device=device)
    

    if opt.find_best_frame:
        i = find_best_frame(source_image, driving_video, opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, dense_motion_network, avd_network, device, kp_source, kp_driving_initial, mode=opt.mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, dense_motion_network, avd_network, device, kp_source, kp_driving_initial, mode=opt.mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, inpainting, dense_motion_network, avd_network, device, kp_source, kp_driving_initial, mode=opt.mode)
    
    # Save the result video
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # Optionally add audio
    if opt.addaudio:
        cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 result_withAudio.mp4'.format(
            opt.result_video, opt.driving_video)
        subprocess.call(cmd, shell=True)

