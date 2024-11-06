import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hdf5_file_path = 'checkpoints/fixed_keypoints.h5'  # Path to your HDF5 file

with h5py.File(hdf5_file_path, 'r') as f:
    if 'keypoints' not in f:
        logging.error("The 'keypoints' dataset is missing in the HDF5 file.")
    else:
        keypoints = f['keypoints']
        logging.info(f"'keypoints' dataset shape: {keypoints.shape}")

        # Check the contents of the first frame
        first_frame = keypoints[:, :, :, 0]
        logging.info(f"Data for first frame (shape {first_frame.shape}):\n{first_frame}")

        # Optionally, check a few random frames
        for frame_idx in [100, 500, 1000]:  # Adjust indices as needed
            if frame_idx < keypoints.shape[3]:
                frame_data = keypoints[:, :, :, frame_idx]
                logging.info(f"Data for frame {frame_idx} (shape {frame_data.shape}):\n{frame_data}")
            else:
                logging.warning(f"Frame index {frame_idx} is out of bounds.")

