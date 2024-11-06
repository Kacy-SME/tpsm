import os
import sys
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import matplotlib
import yaml
import torch
from skimage.transform import resize
import imageio
import h5py  # For loading keypoints from HDF5
from frames_dataset_SLEAP import FramesDataset
from modules.inpainting_network import InpaintingNetwork
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
from train_SLEAP import train
from train_avd import train_avd
from reconstruction_SLEAP import reconstruction

# Adjust the fraction as needed
matplotlib.use('Agg')


def load_keypoints_from_hdf5(hdf5_file, frame_idx):
    with h5py.File(hdf5_file, 'r') as f:
        keypoints = f['tracks'][:]  # Adjust based on your file structure
        keypoints_for_frame = keypoints[:, :, :, frame_idx]
    return keypoints_for_frame


if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "train_avd"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--source_image", type=str, help="Path to the source image")
    parser.add_argument("--driving_video", type=str, help="Path to the driving video")
    parser.add_argument("--keypoint_hdf5", type=str, help="Path to the HDF5 file with keypoints")  # New argument for HDF5 keypoints
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])

    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:'+str(opt.device_ids[0]))
        inpainting.to(cuda_device)

    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    if torch.cuda.is_available():
        dense_motion_network.to(opt.device_ids[0])

    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor()
        if torch.cuda.is_available():
            bg_predictor.to(opt.device_ids[0])

    avd_network = None
    if opt.mode == "train_avd":
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                                 **config['model_params']['avd_network_params'])
        if torch.cuda.is_available():
            avd_network.to(opt.device_ids[0])

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train_params = config['train_params']
        train(config, inpainting, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset, opt.keypoint_hdf5)
    elif opt.mode == 'train_avd':
        print("Training Animation via Disentanglement...")
        train_avd(config, inpainting, bg_predictor, dense_motion_network, avd_network, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        source_image = imageio.imread(opt.source_image)
        source_image = resize(source_image, (256, 256))[..., :3]

        reader = imageio.get_reader(opt.driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in reader]
        reader.close()

        # Load keypoints from HDF5 for each frame
        keypoints_list = []
        for frame_idx in range(len(driving_video)):
            keypoints = load_keypoints_from_hdf5(opt.keypoint_hdf5, frame_idx)
            keypoints_list.append(keypoints)

        # Pass pre-loaded keypoints to reconstruction instead of kp_detector
        reconstruction(config, inpainting, keypoints_list, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset)

