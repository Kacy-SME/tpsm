import torch
import yaml
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork

#Path to checkpoint file
checkpoint_path='checkpoints/wasp_check/wasp-v1 15_02_24_20.27.52/00000099-checkpoint.pth.tar'
config_path='config/wasp-v1.yaml'
def load_model(checkpoint_path, config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint to {device}")

    #Load the config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #Extract necessary configuration parameters for InpaintingNetwork
    generator_params = config['model_params']['generator_params']
    common_params = config['model_params']['common_params']
    dense_motion_params = config['model_params']['dense_motion_params']
    #Initializing components
    inpainting_network = InpaintingNetwork(
        num_channels=common_params['num_channels'],
        block_expansion=generator_params['block_expansion'],
        max_features=generator_params['max_features'],
        num_down_blocks=generator_params['num_down_blocks'],
        multi_mask=common_params['multi_mask']
    ).to(device)

    kp_detector = KPDetector(
            num_tps=common_params['num_tps']
    ).to(device)
    
    dense_motion_network = DenseMotionNetwork(
            block_expansion=dense_motion_params['block_expansion'],
            max_features=dense_motion_params['max_features'],
            num_blocks=dense_motion_params['num_blocks'],
            scale_factor=dense_motion_params['scale_factor'],
            num_tps=common_params['num_tps'],
            num_channels=common_params['num_channels'],
            bg=common_params['bg'],
            multi_mask=common_params['multi_mask'],
            kp_variance=0.01 #Assuming a default value; adjust as necessary

    ).to(device)

    #Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    #Load states into components
    inpainting_network.load_state_dict(checkpoint['inpainting_network'])
    print("Inpainting network loaded successfully.")
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])

    print("Model components loaded successfully")

def check_checkpoint_contents(checkpoint_path):
    #Check if CUDA is available
    if torch.cuda.is_available():
        #Specify CUDA device
        device = torch.device("cuda")
        print("Loading checkpoint to CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Loading checkpoint to CPU")

    #Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    #Print out the keys to see what's inside the checkpoint
    print("Checkpoint keys: ", checkpoint.keys())

    # Optionally, print more details about specific parts of the checkpoint
    # For example, to print the size of the model state dict:
    if 'model_state_dict' in checkpoint:
        print("Model state dict keys:", checkpoint['model_state_dict'].keys())
        for key, value in checkpoint['model_state_dict'].items():
            print(f"{key}: {value.size()}")
    else:
        print("No model_state_dict found in checkpoint.")




if __name__ == "__main__":
    check_checkpoint_contents(checkpoint_path)
    load_model(checkpoint_path, config_path)
