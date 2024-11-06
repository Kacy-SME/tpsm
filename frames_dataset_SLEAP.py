import os
import h5py
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from skimage.transform import resize
import numpy as np
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
import glob
from functools import partial

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith(('.gif', '.mp4', '.mov')):
        video = mimread(name, memtest=False)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions %s" % name)

    return video_array

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, keypoints_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.keypoints_dir = keypoints_dir  # Directory containing HDF5 files for keypoints
        self.videos = os.listdir(root_dir)
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            if len(self.videos) == 1:
                print("Training with only one video, no test set.")
                train_videos = self.videos
            else:
                print("Use random train-test split.")
                train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)
    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)

        # Load video frames
        video_array = read_video(path, frame_shape=self.frame_shape)
        num_frames = len(video_array)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
        video_array = video_array[frame_idx]

        # Extract video name without file extension for keypoints lookup
        video_name = os.path.splitext(name)[0]

        # Load corresponding HDF5 keypoints file
        keypoints_path = os.path.join(self.keypoints_dir, f"{video_name}.h5")
        if not os.path.exists(keypoints_path):
            raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
    
        with h5py.File(keypoints_path, 'r') as kp_file:
            keypoints = kp_file['keypoints'][...]  # Ensure key 'keypoints' matches your HDF5 structure

        # Apply any transformations if needed
        if self.transform is not None:
            video_array = self.transform(video_array)

        # Prepare output dictionary
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['keypoints'] = keypoints  # Add keypoints to the output
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            out['keypoints'] = keypoints  # Add keypoints to the output

        return out

