from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.util_demo import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util_demo import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math
import logging

class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels, 
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask
        # Initialize a layer to reduce channels from 84 to 3 if needed
        self.conv_to_3 = nn.Conv2d(in_channels=84, out_channels=3, kernel_size=1)
 
        self.hourglass = Hourglass(block_expansion=block_expansion, 
                                   in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
                                   max_features=max_features, 
                                   num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], 13, kernel_size=(7, 7), padding=(3, 3))

        if multi_mask:
            up = []
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4
            
            channel = [hourglass_output_size[-1] // (2 ** i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))
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
        self.conv_reduce = nn.Conv2d(in_channels=61, out_channels=45, kernel_size=1)

    def pad_tensor(self, tensor, target_size, target_channels=84):
        """
        Pads or crops the given tensor to match the target batch size.
        If target_channels is provided, the tensor is padded or cropped to match target_channels.
        """
        # Adjust the number of channels to exactly `target_channels`
        current_channels = tensor.size(1)
        if current_channels != target_channels:
            if current_channels > target_channels:
                tensor = tensor[:, :target_channels]  # Crop extra channels
            else:
                # Pad with zeros if there are too few channels
                padding_channels = target_channels - current_channels
                channel_padding = torch.zeros((tensor.size(0), padding_channels, *tensor.shape[2:]),
                                           device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor, channel_padding), dim=1)

        # Adjust the batch size to match `target_size`
        current_batch_size = tensor.size(0)
        if current_batch_size < target_size:
            padding_size = target_size - current_batch_size
            batch_padding = torch.zeros((padding_size, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, batch_padding), dim=0)
        elif current_batch_size > target_size:
            tensor = tensor[:target_size]  # Crop extra batches if needed

        return tensor

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        device = source_image.device
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance).to(device)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance).to(device)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
    
        logging.info(f"heatmap shape before concatenation: {heatmap.shape}")
        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        device = source_image.device
        bs, _, h, w = source_image.shape

        kp_1 = kp_driving['fg_kp'].to(device)
        kp_2 = kp_source['fg_kp'].to(device)

        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)

        trans = TPS(mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image).to(device)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(device)
        identity_grid = identity_grid.view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)

        if bg_param is not None:            
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3).to(device), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):
        device = source_image.device
        bs, _, h, w = source_image.shape

        num_transformations = transformations.size(1)
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, num_transformations, 1, 1, 1, 1).to(device)
        source_repeat = source_repeat.view(bs * num_transformations, -1, h, w)
        
        transformations = transformations.view((bs * num_transformations, h, w, 2)).to(device)
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, num_transformations, -1, h, w))
        return deformed
    
    def dropout_softmax(self, X, P):
        drop = (torch.rand(X.shape[0], X.shape[1]) < (1 - P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2], X.shape[3], 1, 1).permute(2, 3, 0, 1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:, 1:, ...] /= (1 - P)
        mask_bool = (drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition
    def forward(self, source_image, kp_driving, kp_source, bg_param=None, dropout_flag=False, dropout_p=0):
        device = self.maps.weight.device
        out_dict = {}

        # Log initial shape and conditionally reduce channels if they are initially 84
        logging.info(f"Initial source_image shape: {source_image.shape}")
        if source_image.size(1) == 84:
            source_image = self.conv_to_3(source_image)  # Reduces to 3 channels
        logging.info(f"Reduced source_image channels to {source_image.size(1)}")

        # Downsample conditionally, and restore to 84 channels if reduced to 3
        if self.scale_factor != 1:
            logging.info(f"Before self.down, shape: {source_image.shape}")
            source_image = self.down(source_image)
            logging.info(f"After self.down, shape: {source_image.shape}")

        # Restore to 84 channels if downsampling reduced channels
        if source_image.size(1) == 3:
            conv_to_84 = nn.Conv2d(in_channels=3, out_channels=84, kernel_size=1).to(device)
            source_image = conv_to_84(source_image)
            logging.info(f"Restored source_image channels to {source_image.size(1)}")

        # Continue with further processing...
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source_image, transformations)

        logging.info(f"Heatmap representation shape: {heatmap_representation.shape}")
        logging.info(f"Deformed source shape before padding: {deformed_source.shape}")

        # Adjust `deformed_source` to match heatmap_representation channels
        deformed_source = self.pad_tensor(deformed_source.view(source_image.size(0), -1, *deformed_source.shape[2:]), 
                                          target_size=heatmap_representation.size(0), target_channels=61)
        logging.info(f"Deformed source shape after padding: {deformed_source.shape}")

        # Flatten deformed source if necessary
        deformed_source = deformed_source.view(deformed_source.size(0), -1, *deformed_source.shape[-2:])
        logging.info(f"Deformed source shape after flattening: {deformed_source.shape}")
        out_dict['deformed_source'] = deformed_source

        # Concatenate `heatmap_representation` and `deformed_source`
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        logging.info(f"Concatenated input shape: {input.shape}")

        # Reduce input to 84 channels before passing to hourglass
        reduction_layer = nn.Conv2d(in_channels=input.size(1), out_channels=84, kernel_size=1).to(device)
        input = reduction_layer(input)
        logging.info(f"Input shape after reduction to 84 channels: {input.shape}")

        # Pass through hourglass
        try:
            prediction = self.hourglass(input)
            if not isinstance(prediction, list):
                prediction = [prediction]  # Convert to list for consistency
            logging.info(f"Hourglass output shape: {prediction[-1].shape}")
        except Exception as e:
            logging.error(f"Error during hourglass forward pass: {e}")
            raise

        # Compute contribution maps
        contribution_maps = self.maps(prediction[-1]).to(device)
        if dropout_flag:
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Create deformation map
        contribution_maps = contribution_maps.unsqueeze(2)  # (B, 13, 1, H, W)
        transformations = transformations[:, :13, ...]
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1).permute(0, 2, 3, 1)
        out_dict['deformation'] = deformation

        # Process occlusion maps with updated index checks
        occlusion_map = []
        logging.info(f"self.occlusion_num: {self.occlusion_num}, self.up_nums: {self.up_nums}")
        logging.info(f"len(self.occlusion): {len(self.occlusion)}, len(prediction): {len(prediction)}")

        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                index = self.up_nums - self.occlusion_num + i
                if 0 <= index < len(prediction):
                    occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[index])))
                else:
                    logging.error(f"Index {index} out of range for prediction with length {len(prediction)}")
                    continue

            if len(prediction) > 0:
                prediction = prediction[-1]
            else:
                raise RuntimeError("No prediction layers available for final upsampling")

            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_index = i + self.occlusion_num - self.up_nums
                if 0 <= occlusion_index < len(self.occlusion):
                    occlusion_map.append(torch.sigmoid(self.occlusion[occlusion_index](prediction)))
                else:
                    logging.error(f"Index {occlusion_index} out of range for occlusion with length {len(self.occlusion)}")
                    continue
        else:
            if len(prediction) > 0 and len(self.occlusion) > 0:
                occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
            else:
                raise RuntimeError("Prediction or occlusion layers are unavailable for processing")

        out_dict['occlusion_map'] = occlusion_map
        return out_dict

