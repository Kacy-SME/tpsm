import h5py
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model_SLEAP import GeneratorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from frames_dataset import DatasetRepeater
import math
import numpy as np

# Padding function
def pad_kp_driving(kp_driving, batch_size):
    num_instances = kp_driving.shape[0]
    remainder = num_instances % batch_size
    if remainder != 0:
        # Calculate how many more instances we need to pad
        padding_needed = batch_size - remainder
        # Create padding with zeros or another value (same shape as kp_driving[0])
        pad_shape = (padding_needed,) + kp_driving.shape[1:]  # Maintain the same shape except for the first dimension
        padding = np.zeros(pad_shape)
        # Concatenate the padding to kp_driving
        kp_driving_padded = np.concatenate((kp_driving, padding), axis=0)
        return kp_driving_padded
    return kp_driving  # No padding needed if divisible

# Load and preprocess keypoints
def load_and_preprocess_keypoints_from_hdf5(hdf5_file, batch_size):
    with h5py.File(hdf5_file, 'r') as f:
        keypoints = f['keypoints'][:]  # Load the keypoints from HDF5
    # Pad the keypoints to fit the batch size
    keypoints = pad_kp_driving(keypoints, batch_size)
    return keypoints

def train(config, inpainting_network, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset, keypoint_hdf5):
    # Extract training parameters
    train_params = config['train_params']
    
    # Load keypoints from HDF5
    keypoints_driving = load_and_preprocess_keypoints_from_hdf5(keypoint_hdf5, batch_size=train_params['batch_size'])
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()), 
         'initial_lr': train_params['lr_generator']}], 
        lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay=1e-4)

    # Load checkpoint if exists
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, 
                                      dense_motion_network=dense_motion_network, bg_predictor=bg_predictor, 
                                      optimizer=optimizer)
        start_epoch += 1
    else:
        start_epoch = 0

    scheduler_optimizer = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)

    # Repeat dataset if specified in config
    if 'num_repeats' in train_params and train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, 
                            num_workers=train_params['dataloader_workers'], drop_last=True)
    # Assuming you have your DataLoader defined as `dataloader`
    data_iter = iter(dataloader)  # Create an iterator from the dataloader
    batch = next(data_iter)       # Get the first batch of data

    # Print the keys in the batch
    print("Keys in the batch:", batch.keys())

    # Optionally, inspect the content of one of the keys (e.g., 'driving')
    print("Driving data shape:", batch['driving'].shape if 'driving' in batch else "No driving data found")
    print("Source data shape:", batch['source'].shape if 'source' in batch else "No source data found")

    # Initialize full generator model (without kp_detector)
    generator_full = GeneratorFullModel(bg_predictor, dense_motion_network, inpainting_network, train_params)

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full).cuda()
    print(f"Number of batches in DataLoader: {len(dataloader)}")
    print(f"log_dir is: {log_dir}")

#begin training
# Checking and using 'name' as 'video' in the training loop
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                if torch.cuda.is_available():
                    x['driving'] = x['driving'].cuda()
                    x['source'] = x['source'].cuda()

                # Replace kp_detector with pre-loaded keypoints
                kp_source = keypoints_driving[:, :, :, 3995]  # Assuming this frame for source keypoints
                for frame_idx in range(x['driving'].shape[2]):
                    kp_driving = keypoints_driving[:, :, :, frame_idx]  # Use pre-loaded keypoints for driving

                    # Convert np.array to tensor and handle NaN values
                    kp_driving_tensor = torch.tensor(np.nan_to_num(kp_driving, nan=0.0)).float()
                    kp_source_tensor = torch.tensor(np.nan_to_num(kp_source, nan=0.0)).float()

                    # Print shapes before reshaping
                    print(f"kp_driving shape before reshaping: {kp_driving_tensor.shape}")
                    print(f"kp_source shape before reshaping: {kp_source_tensor.shape}")
                    # Calculate total elements
                    total_elements = kp_driving_tensor.numel()
                    print(f"Total elements in kp_driving: {total_elements}")
                    # You can retrieve the batch size from the train_params
                    batch_size = train_params['batch_size']

                    # Assuming the second dimension (number of keypoints) is correct:
                    num_keypoints = total_elements // (batch_size * 2)  # Assuming each keypoint has 2 coordinates (x, y)

                    # Reshape to match the expected input for kp2gaussia
                    kp_driving_reshaped = kp_driving_tensor.view(batch_size, num_keypoints, 2)
                    kp_source_reshaped = kp_source_tensor.view(batch_size, num_keypoints, 2)
                    # Check the structure after reshaping
                    print(f"kp_driving reshaped: {kp_driving_reshaped.shape}")
                    print(f"kp_source reshaped: {kp_source_reshaped.shape}")

                    # Pass the reshaped keypoints to the model
                    kp_source_dict = {'fg_kp': kp_source_reshaped}
                    kp_driving_dict = {'fg_kp': kp_driving_reshaped}

                    # Feed keypoints into the model
                    losses_generator, generated = generator_full(x, epoch, kp_source=kp_source_dict, kp_driving=kp_driving_dict)
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    loss.backward()

                    clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type=math.inf)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log the losses
                    losses = {key: value.mean().detach().cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)

            # Scheduler step
            scheduler_optimizer.step()

            # Save model and log epoch
            model_save = {
                'inpainting_network': inpainting_network,
                'dense_motion_network': dense_motion_network,
                'optimizer': optimizer,
            }
            # Debug print for generated output before calling logger
            print(f"Generated keypoints driving (kp_driving) shape: {generated['kp_driving']['fg_kp'].shape}")
            print(f"Generated keypoints source (kp_source) shape: {generated['kp_source']['fg_kp'].shape}")

            # Additional check for keypoints content, inspect a few entries
            print(f"First few kp_driving keypoints: {generated['kp_driving']['fg_kp'][:2]}")
            print(f"First few kp_source keypoints: {generated['kp_source']['fg_kp'][:2]}")

            # Now log epoch after ensuring the keypoints are in the correct shape
            logger.log_epoch(epoch, model_save, inp=x, out=generated)


