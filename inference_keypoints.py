import torch
from modules.model import GeneratorFullModel, Vgg19, ImagePyramide
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.inpainting_network import InpaintingNetwork

#Change number of keypoints based on data
num_keypoints = 10

# Load the state dictionary from the checkpoint file
state_dict = torch.load('/Users/kacy/tpsm/checkpoints/vox.pth.tar', map_location='cpu')

# Print checkpoint keys
print("Checkpoint keys:")
for key in state_dict.keys():
    print(key)

# Remove the unexpected key
if 'avd_network' in state_dict:
    del state_dict['avd_network']

# Modify the state dictionary keys to match the names in your model
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('kp_detector', 'kp_extractor')  # Assuming the state dict uses different naming
    new_state_dict[new_key] = value


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
	num_channels=3, 	#for RBG images
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

# Assuming generator_model is your model instance
# Print model's state dictionary keys
print("\nModel's expected keys:")
for key in generator_model.state_dict().keys():
    print(key)

# Load the adjusted state dictionary into your model
generator_model.load_state_dict(new_state_dict)
print("Loaded state dict keys:", state_dict.keys())

# Modify the state dictionary keys if necessary
new_state_dict = {key.replace('kp_detector', 'kp_extractor'): value for key, value in state_dict.items()}

# Load the adjusted state dictionary into the model
generator_model.load_state_dict(new_state_dict)

# Output the keys expected by the model, useful for debugging
print("Model's expected keys:", [name for name, _ in generator_model.state_dict().items()])

# Instantiate the GeneratorFullModel
generator_model = GeneratorFullModel(kp_extractor=kp_detector,
                                     bg_predictor=bg_predictor,
                                     dense_motion_network=dense_motion_network,
                                     inpainting_network=inpainting_network,
                                     train_params=train_params)

# Load pre-trained weights into the GeneratorFullModel
generator_model.load_state_dict(new_state_dict, strict=False)

# Optionally, print the model and checkpoint keys for debugging
print("Model's expected keys:", [name for name, _ in generator_model.state_dict().items()])
print("Loaded state dict keys:", new_state_dict.keys())

generator_model.eval()  # Set the model to evaluation mode

# Assume you have a function to preprocess your input data
input_data = preprocess_data('/Users/kacy/tpsm/assets/wasp-source.png')

# Perform inference to get keypoints (and other generated outputs)
with torch.no_grad():
    loss_values, generated_outputs = generator_model(input_data, epoch=0)

# The keypoints are likely part of the generated_outputs dictionary
keypoints_source = generated_outputs['kp_source']  # Keypoints from the source image
keypoints_driving = generated_outputs['kp_driving']  # Keypoints from the driving image



