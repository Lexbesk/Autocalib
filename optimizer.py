import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scene.cameras import Camera_Pose  # Assuming this is available
from gaussian_renderer import render_gradio, render_flow  # Adjust imports as needed
from utils_loc.loss_utils import cosine_loss, l1_loss, ssim  # If used in your losses
from optimize_multiframe_droid_batch import iou_loss, flow_to_color  # Adjust imports
from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
import os
from utils_loc.flow_utils import run_flow_on_images, run_tracker_on_images
from PIL import Image
import torchvision.transforms as T

class Optimizer:
    def __init__(self, 
                images, 
                depths, 
                masks, 
                joints, 
                intrinsics,
                init_extrinsics,
                gt_extrinsics,
                gaussians,
                background_color,
                have_depth,
                render_feature_loss,
                direct_feature_loss,
                use_sam_mask,
                vis_features,
                tracking_loss,
                data_path,
                device='cuda'):
        """
        Initialize the Optimizer with dataset-specific information.

        Parameters:
        - images: numpy array or list of images (N, H, W, C)
        - depths: numpy array or list of depth maps (N, H, W)
        - masks: numpy array or list of masks (N, H, W) or None if not using SAM masks
        - joints: numpy array or list of joint angles (N, num_joints)
        - intrinsics: numpy array of camera intrinsics (3, 3)
        - extrinsics: numpy array of camera extrinsics (4, 4)
        - gaussians: GaussianModel object for rendering
        - background_color: torch tensor for background color (3,)
        - use_feature_loss: bool, whether to use feature-based loss
        - use_sam_mask: bool, whether to use SAM masks
        - device: str, device to run computations on ('cuda' or 'cpu')
        """
        self.images = images  # (N, H, W, C)
        self.depths = depths  # (N, H, W)
        self.masks = masks if use_sam_mask else None  # (N, H, W) or None
        self.joints = joints  # (N, num_joints)
        self.intrinsics = torch.tensor(intrinsics, dtype=torch.float32).to(device)
        self.gt_extrinsics = torch.tensor(gt_extrinsics, dtype=torch.float32).to(device)
        self.gaussians = gaussians
        self.background_color = background_color.to(device)
        self.device = device
        self.have_depth = have_depth
        self.render_feature_loss = render_feature_loss
        self.direct_feature_loss = direct_feature_loss
        self.vis_features = vis_features
        self.tracking_loss = tracking_loss
        self.use_sam_mask = use_sam_mask
        self.data_path = data_path
        self.flow_file = os.path.join(data_path, 'tracker_data.pth')
        

        # Extract image dimensions
        self.N, self.H, self.W = images.shape[0], images.shape[1], images.shape[2]
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        self.fovx = 2 * np.arctan(self.W / (2 * fx))
        self.fovy = 2 * np.arctan(self.H / (2 * fy))

        # Preprocess data (convert to tensors and move to device)
        self._preprocess_data()

        # Initialize cameras with shared parameters
        self.cameras = self._initialize_cameras()

        # Load feature extractors if needed
        if self.render_feature_loss or self.direct_feature_loss:
            self._setup_feature_extractors()
        if self.render_feature_loss:
            self._setup_feature_rendering()

    def _preprocess_data(self):
        """Convert images, depths, masks, and joints to tensors and preprocess."""
        self.mujoco_images = []
        self.mujoco_depths = []
        self.mujoco_masks = []
        self.mujoco_feats = []
        self.mujoco_direct_feats = []

        for idx in range(self.N):
            # Images
            img = torch.from_numpy(self.images[idx]).float().to(self.device).permute(2, 0, 1) / 255.0
            self.mujoco_images.append(img)

            # Depths
            depth = torch.from_numpy(self.depths[idx]).float().to(self.device)
            self.mujoco_depths.append(depth)

            # Masks (if applicable)
            if self.use_sam_mask:
                mask = (self.masks[idx] > 0).astype(np.uint8)
                mask_tensor = torch.from_numpy(mask).to(self.device)
                self.mujoco_masks.append(mask_tensor)

            # Feature extraction (if applicable)
            if self.use_feature_loss:
                dino_img = Image.fromarray(self.images[idx])
                dino_img = self.dino_transform(dino_img)[:3].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feats = self.dinov2.forward_features(dino_img)["x_norm_patchtokens"][0]
                self.mujoco_direct_feats.append(feats)
                feats_np = feats.cpu().numpy()
                feats_hwc = feats_np.reshape(
                    (self.target_H // self.dinov2.patch_size, self.target_W // self.dinov2.patch_size, -1)
                )
                self.mujoco_feats.append(torch.from_numpy(feats_hwc).float().to(self.device))

        self.mujoco_images = torch.stack(self.mujoco_images)
        self.mujoco_depths = torch.stack(self.mujoco_depths)
        if self.use_sam_mask:
            self.mujoco_masks = torch.stack(self.mujoco_masks)
        if self.use_feature_loss:
            self.mujoco_feats = torch.stack(self.mujoco_feats)
            self.mujoco_direct_feats = torch.stack(self.mujoco_direct_feats)

    def _setup_feature_rendering(self):
        """Initialize feature extractors for use_feature_loss."""
        self.my_feat_decoder = skip_feat_decoder(32).to(self.device)
        decoder_path = os.path.join(self.gaussians.model_path, 'point_cloud', 'pose_conditioned_iteration_4000', 'feat_decoder.pth')
        self.my_feat_decoder.load_state_dict(torch.load(decoder_path))

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        # Assuming target_H and target_W are derived from resizing logic; adjust as needed
        self.target_H, self.target_W = 800, 800  # Placeholder; match your resizing logic
        
    def _setup_feature_extractors(self):
        """Initialize feature extractors for use_feature_loss."""
        self.my_feat_decoder = skip_feat_decoder(32).to(self.device)
        decoder_path = os.path.join(self.gaussians.model_path, 'point_cloud', 'pose_conditioned_iteration_4000', 'feat_decoder.pth')
        self.my_feat_decoder.load_state_dict(torch.load(decoder_path))

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        # Assuming target_H and target_W are derived from resizing logic; adjust as needed
        self.target_H, self.target_W = 800, 800 
        
    def prepare_flow(self):
        if not os.path.exists(self.flow_file):
            mujoco_images = []
            for c_idx in range(self.N):
                mujoco_image = self.images[c_idx]
                mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(self.device, non_blocking=True)
                mujoco_images.append(mujoco_tensor)
            mujoco_images_target = torch.stack(mujoco_images)
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device, non_blocking=True)
            fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target) # T-1 2 H W, T-1 H W
            fwd_valids = fwd_valids.unsqueeze(1)
            data_to_save = {
                'fwd_flows': fwd_flows.detach().cpu(),
                'fwd_valids': fwd_valids.detach().cpu()
            }
            # Save the dictionary to the file
            torch.save(data_to_save, self.flow_file)
        data_loaded = torch.load(self.flow_file)
        self.fwd_flows = data_loaded['fwd_flows']
        self.fwd_valids = data_loaded['fwd_valids']

    def _initialize_cameras(self):
        """Create Camera_Pose objects with shared parameters across frames."""
        cameras = []
        for idx in range(self.N):
            joint_angles = torch.tensor(self.joints[idx], dtype=torch.float32).to(self.device)
            camera = Camera_Pose(
                self.extrinsics.clone().detach(),
                self.fovx,
                self.fovy,
                self.W,
                self.H,
                joint_pose=joint_angles,
                zero_init=True
            ).to(self.device)

            if first_camera is None:
                first_camera = camera
                # Load optimal parameters if available (assuming Optimal_param is accessible)
                # Modify this based on how you want to handle Optimal_param
            else:
                # Share parameters with the first camera
                for name, param in first_camera.named_parameters():
                    if name in camera._parameters:
                        camera._parameters[name] = param
            cameras.append(camera)
        return cameras

    def optimize(self, optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
        """
        Perform optimization on camera parameters and/or joint angles.

        Parameters:
        - optimize_camera: bool, whether to optimize camera parameters
        - optimize_joints: bool, whether to optimize joint angles
        - camera_lr: float, learning rate for camera parameters
        - joints_lr: float, learning rate for joint angles
        - optimization_steps: int, number of optimization iterations
        - fwd_flows: torch tensor, forward flows (T-1, 2, H, W)
        - fwd_valids: torch tensor, flow validity masks (T-1, 1, H, W)

        Returns:
        - world_view_transform: numpy array, optimized camera transformation
        - valid_optimization: bool, whether optimization was successful
        """
        # Move flow data to device
        fwd_flows = fwd_flows.to(self.device)
        fwd_valids = fwd_valids.to(self.device)

        # Setup optimizers
        optimizers = []
        if optimize_camera:
            # All cameras share parameters, so optimize the first camera's parameters
            optimizer = torch.optim.Adam(self.cameras[0].parameters(), lr=camera_lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            optimizers.append({'optimizer': optimizer, 'scheduler': scheduler})
        if optimize_joints:
            all_joint_poses = [camera.joint_pose for camera in self.cameras]
            optimizer = torch.optim.Adam(all_joint_poses, lr=joints_lr)
            optimizers.append({'optimizer': optimizer, 'scheduler': None})

        # Optimization loop
        tracking_loss = True  # Adjust based on your needs
        for step in range(optimization_steps):
            for opt in optimizers:
                opt['optimizer'].zero_grad()

            # Render outputs
            gaussian_tensors, gaussian_depths, gaussian_masks, gaussian_feats, gaussian_flows = [], [], [], [], []
            for i, camera in enumerate(self.cameras):
                output = render_gradio(camera, self.gaussians, self.background_color, render_features=self.use_feature_loss)
                gaussian_tensors.append(output['render'])
                gaussian_depths.append(torch.nan_to_num(output['depth']))
                thresholded = torch.sigmoid((output['render'] - 0.01) * 500)
                gaussian_masks.append(thresholded.max(dim=0)[0])

                if self.use_feature_loss:
                    feat = output['render_feat'].permute(0, 3, 1, 2)
                    dino_feat = self.my_feat_decoder(feat).permute(0, 2, 3, 1).squeeze()
                    gaussian_feats.append(dino_feat)

                if tracking_loss and i > 0:
                    flow_output = render_flow([self.cameras[i], self.cameras[i-1]], self.gaussians, self.background_color)
                    flow_2d = flow_output["flow"].squeeze().reshape((self.H, self.W, 2)).permute(2, 0, 1)
                    gaussian_flows.append(flow_2d)

            # Stack rendered outputs
            gaussian_tensors = torch.stack(gaussian_tensors)
            gaussian_depths = torch.stack(gaussian_depths)
            gaussian_masks = torch.stack(gaussian_masks)
            if self.use_feature_loss:
                gaussian_feats = torch.stack(gaussian_feats)
            if tracking_loss and gaussian_flows:
                gaussian_flows = torch.stack(gaussian_flows)

            # Compute losses (adapt these based on your specific loss terms)
            l2_diff = F.mse_loss(self.mujoco_images, gaussian_tensors).mean()
            depth_diff = F.mse_loss(self.mujoco_depths, gaussian_depths).mean() * 0.1

            flow_loss = 0
            if tracking_loss and gaussian_flows.numel() > 0:
                flow_diff = (gaussian_flows - fwd_flows).abs()
                flow_diff[:, 0] /= self.W
                flow_diff[:, 1] /= self.H
                motion_idx = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.1).sum(dim=(1, 2, 3)) > 10000
                motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
                flow_loss = (flow_diff * fwd_valids * motion_idx).sum() / fwd_valids.sum() * 1000

            iou_loss_motion = iou_loss(gaussian_masks[:-1], (self.mujoco_depths[:-1] < 0.7).float()) * 0.01 if tracking_loss else 0

            total_loss = l2_diff + depth_diff + flow_loss + iou_loss_motion

            if total_loss == 0:
                continue

            total_loss.backward()
            for opt in optimizers:
                opt['optimizer'].step()
                if opt['scheduler']:
                    opt['scheduler'].step()

        # Determine optimization validity
        valid_optimization = flow_loss < 0.5 if flow_loss > 0 else True

        # Return optimized camera transformation
        world_view_transform = self.cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy()
        return world_view_transform, valid_optimization
