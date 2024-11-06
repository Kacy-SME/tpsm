from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math

class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels, 
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_tps+1) + num_tps*5+1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], 13, kernel_size=(7, 7), padding=(3, 3))


        if multi_mask:
            up = []
            self.up_nums = int(math.log(1/scale_factor, 2))
            self.occlusion_num = 4
            
            channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1]//(2**(i+1)))
            occlusion = []
            
            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance
        # Declare Conv2d layer to reduce heatmap_representation channels
        self.conv_reduce = nn.Conv2d(in_channels=61, out_channels=45, kernel_size=1)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        # Print out the contents and structure of kp_driving
        print("kp_driving type:", type(kp_driving))
        if isinstance(kp_driving, dict):
            print("kp_driving keys:", kp_driving.keys())
            if 'fg_kp' not in kp_driving:
                print("Warning: 'fg_kp' key is not found in kp_driving.")
        else:
            print("kp_driving does not seem to be a dictionary. Check its structure:", kp_driving)
        # Add debug prints to inspect the keypoints being passed
        print(f"kp_driving['fg_kp'] shape: {kp_driving['fg_kp'].shape}")
        print(f"kp_source['fg_kp'] shape: {kp_source['fg_kp'].shape}")

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        # Debugging statements to inspect keypoints
        print(f"kp_1 shape before view: {kp_1.shape}")
        print(f"kp_2 shape before view: {kp_2.shape}")
        
        # Print after reshaping for verification
        print(f"Reshaped kp_1 shape: {kp_1.shape}")  # Should be [16, ...]
        print(f"Reshaped kp_2 shape: {kp_2.shape}")  # Should be [16, ...]

        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)
        # Debugging after view
        print(f"kp_1 shape after view: {kp_1.shape}")
        print(f"kp_2 shape after view: {kp_2.shape}")
        
        trans = TPS(mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        if not (bg_param is None):            
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        print(f"identity_grid shape: {identity_grid.shape}")
        print(f"driving_to_source shape: {driving_to_source.shape}")
        print(f"transformations shape before return: {transformations.shape}")

        return transformations

    def create_deformed_source_image(self, source_image, transformations):
        bs, _, h, w = source_image.shape
        # Dynamically calculate the number of transformations from the transformations tensor
        num_transformations = transformations.size(1)  # Dynamically extract the number of transformations (e.g., 13)

        # Adjust source_repeat for the actual number of transformations (num_tps + 1)
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, num_transformations, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * num_transformations, -1, h, w)
        
        # Debugging print statements
        print(f"Batch size (bs): {bs}")
        print(f"Number of transformations (dynamic): {num_transformations}")
        print(f"Transformations tensor size: {transformations.size()}")
        print(f"Source image tensor size: {source_image.size()}")
        print(f"Expected reshape size: {(bs * num_transformations, h, w, 2)}")

        # Check the total number of elements
        total_elements = transformations.numel()
        print(f"Total elements in transformations: {total_elements}")

        # Calculate the expected number of elements dynamically
        expected_num_elements = bs * num_transformations * h * w * 2  # Factor of 2 for the coordinates
        print(f"Expected total elements for reshape: {expected_num_elements}")

        if total_elements == expected_num_elements:
            # If the number of elements matches, reshape as expected
            transformations = transformations.view((bs * num_transformations, h, w, 2))
        else:
            # Print mismatch and raise an error if there's a discrepancy
            print(f"Mismatch in expected elements: {expected_num_elements} vs actual: {total_elements}")
            raise ValueError(f"Transformation tensor size mismatch: expected {expected_num_elements}, got {total_elements}")

        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, num_transformations, -1, h, w))  # Use dynamic num_transformations
        return deformed
    
    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0], X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2], X.shape[3], 1, 1).permute(2, 3, 0, 1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:, 1:, ...] /= (1-P)
        mask_bool = (drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition

    def forward(self, source_image, kp_driving, kp_source, bg_param=None, dropout_flag=False, dropout_p=0):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict['deformed_source'] = deformed_source

        # Print out the sizes of the tensors before concatenation
        print(f"heatmap_representation shape: {heatmap_representation.shape}")
        print(f"deformed_source shape: {deformed_source.shape}")
    
        # Ensure heatmap_representation has the correct batch size (matching bs)
        if heatmap_representation.shape[0] != bs:
            heatmap_representation = heatmap_representation.view(bs, -1, h, w)  # Adjust batch size to match bs
    
        # Print adjusted shape to verify
        print(f"Adjusted heatmap_representation shape: {heatmap_representation.shape}")

        # Move the existing conv layer to the same device as the input
        device = heatmap_representation.device
        self.conv_reduce = self.conv_reduce.to(device)

        # Apply the convolution using the pre-defined conv layer
        heatmap_representation = self.conv_reduce(heatmap_representation)

        # Check if the shapes are compatible for concatenation
        deformed_source = deformed_source.view(bs, -1, h, w)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        
        # Debugging print statement to check input
        print(f"Input shape to hourglass: {input.shape}")
        prediction = self.hourglass(input, mode=1)
    
        contribution_maps = self.maps(prediction[-1])
        if dropout_flag:
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the K+1 transformations (Eq 6 in the paper)
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        # Debugging: print the shapes before multiplication
        print(f"transformations shape: {transformations.shape}")  # Expected: [16, 13, 64, 64, 2]
        print(f"contribution_maps shape: {contribution_maps.shape}")  # Check if this is [16, 13, 1, 64, 64]

        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation  # Optical Flow

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))

        out_dict['occlusion_map'] = occlusion_map  # Multi-resolution Occlusion Masks
        return out_dict

