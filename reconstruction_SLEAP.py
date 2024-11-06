import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
import h5py  # For loading keypoints from HDF5


# Function to load keypoints from the HDF5 file
def load_keypoints_from_hdf5(hdf5_file, frame_idx):
    with h5py.File(hdf5_file, 'r') as f:
        keypoints = f['tracks'][:]  # Adjust based on your file structure
        keypoints_for_frame = keypoints[:, :, :, frame_idx]
    return keypoints_for_frame


def reconstruction(config, inpainting_network, keypoints_list, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    
    loss_list = []

    inpainting_network.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()

            # Load keypoints for the source frame (assume frame 0 is the source frame)
            kp_source = keypoints_list[0]

            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]

                # Load keypoints for the current driving frame
                kp_driving = keypoints_list[frame_idx]

                bg_params = None
                if bg_predictor:
                    bg_params = bg_predictor(source, driving)

                # Perform dense motion and inpainting with pre-loaded keypoints
                dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param=bg_params, 
                                                    dropout_flag=False)
                out = inpainting_network(source, dense_motion)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                   driving=driving, out=out)
                visualizations.append(visualization)
                loss = torch.abs(out['prediction'] - driving).mean().cpu().numpy()
                
                loss_list.append(loss)

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

    print("Reconstruction loss: %s" % np.mean(loss_list))

