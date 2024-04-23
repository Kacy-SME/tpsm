import torch
from modules.model import GeneratorFullModel, Vgg19, ImagePyramide
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork

# Load the state dictionary from the checkpoint file
checkpoint = torch.load('/Users/kacy/tpsm/checkpoints/vox.pth.tar', map_location='cpu')
# Print out the keys in the checkpoint file
print("Keys in checkpoint:", checkpoint.keys())

# Load state dictionaries from the checkpoint into your model components
kp_detector.load_state_dict(checkpoint['kp_detector'])
inpainting_network.load_state_dict(checkpoint['inpainting_network'])
dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
bg_predictor.load_state_dict(checkpoint['bg_predictor'])


state_dict = checkpoint['state_dict']

# Print checkpoint keys
print("Checkpoint keys:")
for key in state_dict.keys():
    print(key)

# Instantiate necessary components for GeneratorFullModel
kp_extractor = KPDetector(num_tps=num_keypoints)
bg_predictor = BGMotionPredictor()
dense_motion_network = DenseMotionNetwork(
        block_expansion=32,
        num_blocks=4,
        max_features=256,
        num_tps=10,
        num_channels=3
)
inpainting_network = InpaintingNetwork(
        num_channels=3,         #for RBG images
        block_expansion=32,        # Example value
        max_features=256,          # Example maximum features
        num_down_blocks=4          # Example number of down-sampling blocks
)
train_params = {
    'bg_start': 10,
    'lr': 0.001,  # Learning rate
    'batch_size': 16,  # Batch size
    'num_epochs': 100,  # Number of training epochs
    'loss_weights': {
        'perceptual': [0.1, 0.2, 0.3, 0.4],  # Weights for different levels of perceptual loss
        'equivariance_value': 0.1,  # Weight for equivariance loss
        'warp_loss': 0.1,  # Weight for warp loss
        'bg': 0.1  # Weight for background loss
    },
    'scales': [1, 0.5, 0.25],  # Different scales for multi-scale processing
    'dropout_epoch': 10,  # Epoch after which to start dropout
    'dropout_maxp': 0.5,  # Maximum dropout probability
    'dropout_inc_epoch': 5,  # Dropout increment epoch
    'dropout_startp': 0.1,  # Dropout starting probability
    'transform_params': {  # Parameters for transformation, used in equivariance loss calculation
        # ... You will need to add the specific parameters here
    }
    # ... potentially other parameters as needed
}

# Initialize your model
generator_model = GeneratorFullModel(kp_extractor, bg_predictor, dense_motion_network, inpainting_network, train_params)

# Print model's expected keys
print("Model's expected keys:")
model_keys = set(key for key, _ in generator_model.state_dict().items())
for key in model_keys:
    print(key)

# Modify the state dictionary keys if necessary
# This is an example, adjust the 'prefix_to_remove' and 'prefix_to_add' as needed
prefix_to_remove = 'module.'
prefix_to_add = ''
new_state_dict = {key.replace(prefix_to_remove, prefix_to_add): value for key, value in state_dict.items() if key.startswith(prefix_to_remove)}

# Check for keys that didn't match up and weren't renamed, print them out for manual inspection
unmatched_keys = model_keys - set(new_state_dict.keys())
if unmatched_keys:
    print("Unmatched keys that were not found in the checkpoint state_dict:")
    for key in unmatched_keys:
        print(key)

# Load the adjusted state dictionary into your model
generator_model.load_state_dict(new_state_dict)
print("Loaded state dict keys:", state_dict.keys())

# If you're sure all keys are accounted for, you can use strict=False to ignore non-matching keys
generator_model.load_state_dict(new_state_dict, strict=False)
