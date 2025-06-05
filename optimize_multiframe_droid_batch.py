import sys
import tempfile

"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_reconstruct.py --model_path output/[path_to_your_model_directory]
python gradio_app_reconstruct_rh20t.py --model_path assets/ur5
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python gradio_app_reconstruct_rh20t_multiple_view.py --model_path output/
python optimize_multiframe.py --model_path output/universal_robots_ur5e_robotiq_experiment/
python optimize_multiframe.py --model_path output/franka_emika_panda_complement1
python optimize_multiframe.py --model_path output/franka_emika_panda6
python optimize_multiframe.py --model_path output/franka_
python optimize_multiframe_droid_batch.py --model_path output/franka_fr3_2f85_highres_finetune_0
This file is for UR5
"""


import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir

import gradio as gr

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

import numpy as np
import mujoco
from utils_loc.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from tqdm import tqdm

from video_api import initialize_gaussians
from gaussian_renderer import render, render_gradio, render_flow
from scene.cameras import Camera_Pose, Camera
from utils_loc.flow_utils import run_flow_on_images, run_tracker_on_images
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss
import pickle



def rotation_matrix_to_quaternion(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    return R.from_quat(quat).as_matrix()

gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()

use_feature_loss = False
if use_feature_loss:
    my_feat_decoder = skip_feat_decoder(32).cuda()
    decoder_dict_path = os.path.join(gaussians.model_path, 'point_cloud', 'pose_conditioned_iteration_4000', 'feat_decoder.pth')
    my_feat_decoder.load_state_dict(torch.load(decoder_dict_path))

background_color = torch.zeros((3,)).cuda()
# make it orange
# background_color[0] = 1.0
# background_color[1] = 0.5
example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'model.xml'))
data = mujoco.MjData(model)
n_joints = model.njnt

def flow_to_color(flow):
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0, 1]
    mag = np.sqrt(dx**2 + dy**2)
    mag_max = np.max(mag)
    mag_norm = mag / mag_max if mag_max > 0 else mag
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=2)
    rgb = mcolors.hsv_to_rgb(hsv)
    return np.clip(rgb, 0, 1)  # Clip to [0, 1]

def iou_loss(pred, target, smooth=1e-6):
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)  # (batch_size, H*W)
    target = target.view(batch_size, -1)
    intersection = (pred * target).sum(dim=1)  # Sum over pixels per batch
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou).mean()  # Average loss over batch

def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, powerful_optimize_dropdown, noise_input, num_inits_input):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    mujoco_feats = []
    mujoco_direct_feats = []
    first_camera = None  # Track the first camera to share its parameters
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs('inspection_features', exist_ok=True)
    if use_feature_loss:
        print("Loading DINOv2 model...")
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        
        if use_feature_loss:
            dino_image = mujoco_image
            # print(dino_image.shape, 'dino_image')
            dino_image = Image.fromarray(dino_image)
            dino_image.save(f'inspection_features/dino_image_{c_idx}.png')
            dino_image = resize_image(dino_image, 800)
            dino_image = dino_transform(dino_image)[:3].unsqueeze(0)
            dino_image, target_H, target_W = interpolate_to_patch_size(dino_image, dinov2.patch_size)
            dino_image = dino_image.cuda()
            with torch.no_grad():
                features = dinov2.forward_features(dino_image)["x_norm_patchtokens"][0]
            mujoco_direct_feats.append(features)
            features = features.cpu().numpy()
            # print(features.shape, 'features')
            features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            features_hwc = torch.from_numpy(features_hwc).float().cuda()
            # print(features_hwc.shape, 'features_hwc')
            mujoco_feats.append(features_hwc)
        
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        mujoco_mask = mask_list[c_idx]
        mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
        mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
        # print(mujoco_mask_tensor.shape, 'mujoco_mask_tensor', mujoco_mask_tensor.sum())
        mujoco_masks.append(mujoco_mask_tensor)
        mujoco_depth = depth_list[c_idx]
        mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
        mujoco_depth_tensor = mujoco_depth_tensor
        mujoco_depths.append(mujoco_depth_tensor)
        # Only perturb parameters that are being optimized
        # print("gaussian_params", gaussian_params)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
            if cam_id in Optimal_param:
                Optimal_Camera_Param = Optimal_param[cam_id]
                first_camera.w.data = Optimal_Camera_Param['w'].to(camera.device)
                first_camera.v.data = Optimal_Camera_Param['v'].to(camera.device)
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)

    optimizers = []
    if optimize_camera:
        optimizers.append(torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr))
    if optimize_joints:
        optimizers.append(torch.optim.Adam(all_joint_poses, lr=joints_lr))

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        for optimizer in optimizers:
            optimizer.zero_grad()

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_feats = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            # if i not in Optimize_list:
            #     print(i)
            #     i += 1
            #     continue
            # camera.joint_pose = joint_pose
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features = use_feature_loss)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features = use_feature_loss)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
            elif UR5:
                output = render_gradio(camera, gaussians, background_color, render_features = use_feature_loss)
            
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
            if i > 0 and i < 7 and step % 60 == 0:
                flow_2d_np = flow_2d.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                new_image_np = new_image.permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np = flow_to_color(flow_2d_np)
                step_grid = 10
                H_f, W_f = flow_2d_np.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_np[y, x, 0]
                dy = flow_2d_np[y, x, 1]

                # Step 4: Create the figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(new_image_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(flow_color_np)
                axes[1].set_title('Flow Color Map')
                axes[1].axis('off')

                axes[2].imshow(new_image_np)
                axes[2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2].set_title('Image with Flow Arrows')
                axes[2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features/flow_visualization{i}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
                
                points = gaussian_2d_pos_curr.detach().cpu().numpy()  # Shape: [N, 2]
                x = points[:, 0]
                y = points[:, 1]

                # Get image dimensions
                H_f, W_f = new_image_np.shape[:2]
                # print(new_image_np.shape, 'new image np shape')
                # Create mask
                points_int = np.round(points).astype(int)
                mask = np.zeros((H_f, W_f), dtype=np.uint8)
                valid = (0 <= points_int[:, 0]) & (points_int[:, 0] < W_f) & (0 <= points_int[:, 1]) & (points_int[:, 1] < H_f)
                mask[points_int[valid, 1], points_int[valid, 0]] = 1

                # Create figure with two subplots
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Subplot 1: Image with points
                axes[0].imshow(new_image_np)
                axes[0].scatter(x, y, c='red', s=1)
                axes[0].set_title('Image with Points')
                axes[0].axis('off')

                # Subplot 2: Image with mask overlay
                axes[1].imshow(new_image_np)
                axes[1].imshow(mask, cmap='jet', alpha=0.5)
                axes[1].set_title('Image with Mask Overlay')
                axes[1].axis('off')
                plt.savefig(f'inspection_features/points_and_mask_visualization{i}.png', dpi=300)
                plt.close()

            if use_feature_loss:
                rendered_feat = output['render_feat'].permute(0, 3, 1, 2)
                dino_feat = my_feat_decoder(rendered_feat).permute(0, 2, 3, 1).squeeze()
                gaussian_feats.append(dino_feat)
            
            if step == 0:
                # print(f'-----------------------------------saving pcd from {i}')
                os.makedirs("pcd", exist_ok=True)
                np.save(f"pcd/means3D_{i}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            os.makedirs('tmp', exist_ok=True)
            mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if use_feature_loss:
            gaussian_feats = torch.stack(gaussian_feats)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        print(gaussian_flows.shape, 'gaussian flows')
        
        
        
        print(step, 'step')
        if tracking_loss and step == 0:
            mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            mujoco_depths_target = torch.stack(mujoco_depths)
            print(mujoco_images_target.shape, 'mujoco images target shape')
            
            data_file = os.path.join(DATA_PATH, 'tracker_data.pth')
            if os.path.exists(data_file):
                # Load the data from the file
                data_loaded = torch.load(data_file)
                fwd_flows = data_loaded['fwd_flows']
                fwd_valids = data_loaded['fwd_valids']
                fwd_flows = fwd_flows.to(device)
                fwd_valids = fwd_valids.to(device)
                if fwd_flows.shape != gaussian_flows.shape:
                    print('shape changed, recalculating flow: ', fwd_flows.shape, gaussian_flows.shape)
                    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
                    # Run cotracker
                    fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target) # T-1 2 H W, T-1 H W
                    fwd_valids = fwd_valids.unsqueeze(1)
                    # Prepare the data by moving tensors to CPU and storing in a dictionary
                    data_to_save = {
                        'fwd_flows': fwd_flows.detach().cpu(),
                        'fwd_valids': fwd_valids.detach().cpu()
                    }
                    # Save the dictionary to the file
                    torch.save(data_to_save, data_file)
            else:
                cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
                # Run cotracker
                fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target) # T-1 2 H W, T-1 H W
                fwd_valids = fwd_valids.unsqueeze(1)
                # Prepare the data by moving tensors to CPU and storing in a dictionary
                data_to_save = {
                    'fwd_flows': fwd_flows.detach().cpu(),
                    'fwd_valids': fwd_valids.detach().cpu()
                }
                # Save the dictionary to the file
                torch.save(data_to_save, data_file)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 2)
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            depth_mask = (mujoco_depths_target[:-1] > 0.5)
            depth_near_mask1 = (mujoco_depths_target < 0.7)
            depth_near_mask = (mujoco_depths_target[:-1] < 0.7)
            depth_near_mask = depth_near_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            motion_depth_near_mask = motion_mask * depth_near_mask
            print(mujoco_depths_target)
            print(depth_mask)
            depth_mask = depth_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            
            print(depth_mask.shape, 'depth mask shape')
            # print(motion_mask)
            print(motion_mask.shape, 'motion mask shape')
            
            
            
        if tracking_loss and step % 40 == 0:
            for i in range(1, mujoco_images_target.shape[0]):
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                
                flow_2d_np_render = gaussian_flows[i-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                step_grid = 10
                new_image_render = gaussian_tensors[i].permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np_render = flow_to_color(flow_2d_np_render)
                H_f, W_f = flow_2d_np_render.shape[:2]
                x_render, y_render = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx_render = flow_2d_np_render[y_render, x_render, 0]
                dy_render = flow_2d_np_render[y_render, x_render, 1]
                
                flow_2d_mujoco = fwd_flows[i-1].permute(1, 2, 0).detach().cpu().numpy()
                new_image_np = (mujoco_images_target[i] / 255).permute(1, 2, 0).detach().cpu().numpy()
                flow_color_mujoco = flow_to_color(flow_2d_mujoco)
                step_grid = 10
                H_f, W_f = flow_2d_mujoco.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_mujoco[y, x, 0]
                dy = flow_2d_mujoco[y, x, 1]
                
                mask_i = motion_mask[i-1, 0, :, :].float().cpu().numpy()
                depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
                motion_depth_near_mask_i = motion_depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                print(flow_valid_mask_i.min(), flow_valid_mask_i.max(), 'flow valid mask min max')
                

                axes[0, 0].imshow(new_image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(flow_color_mujoco)
                axes[0, 1].set_title('Flow Color Map')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(new_image_np)
                axes[0, 2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[0, 2].set_title('Image with Flow Arrows')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(new_image_render)
                axes[1, 0].set_title('Rendered Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(flow_color_np_render)
                axes[1, 1].set_title('Rendered Flow Color Map')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(new_image_render)
                axes[1, 2].quiver(x, y, dx_render, dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[1, 2].set_title('Rendered Flow Arrows')
                axes[1, 2].axis('off')

                axes[2, 0].imshow(new_image_np)
                axes[2, 0].imshow(motion_depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 0].set_title('Motion Mask')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(new_image_np)
                axes[2, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 1].set_title('Depth Mask')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(new_image_np)
                axes[2, 2].quiver(x, y, dx - dx_render, dy - dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2, 2].set_title('Image with Flow Arrows')
                axes[2, 2].axis('off')
                
                axes[3, 0].imshow(new_image_np)
                axes[3, 0].imshow(depth_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 0].set_title('Depth Mask')
                axes[3, 0].axis('off')

                axes[3, 1].imshow(new_image_np)
                axes[3, 1].imshow(depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 1].set_title('Valid Mask')
                axes[3, 1].axis('off')

                axes[3, 2].imshow(new_image_np)
                axes[3, 2].imshow(flow_valid_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 2].set_title('Flow valid mask')
                axes[3, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features/super_flow_visualization{i}_{cam_name}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
    
        if tracking_loss:
            # target_mask = (self.input_mask_torch_orig_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float()

            flow_loss = (gaussian_flows - fwd_flows).abs()
            
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss_motion = (flow_loss * motion_mask * fwd_valids).sum() / (motion_mask * fwd_valids).sum()
            flow_loss = (flow_loss * fwd_valids).sum() / fwd_valids.sum()
            
            # # Note that this is different from the normal balanced mask loss
            # if target_mask.sum() > 0:
            #     if (target_mask * obj_flow_valid_mask).sum() > 0:
            #         masked_flow_loss = (flow_loss * obj_flow_valid_mask * target_mask).sum() / (target_mask * obj_flow_valid_mask).sum()
            #     else:
            #         masked_flow_loss = 0.
            #     if (1 - target_mask).sum() > 0:
            #         masked_flow_loss_empty = (flow_loss * (1 - target_mask)).sum() / (1 - target_mask).sum()
            #     else:
            #         masked_flow_loss_empty = 0.
            #     loss = loss + 10000. * step_ratio * (masked_flow_loss + masked_flow_loss_empty) / self.opt.batch_size
        
        if use_feature_loss:
            mujoco_feats_target = torch.stack(mujoco_feats)
            mujoco_direct_feats_target = torch.stack(mujoco_direct_feats)
            features_final = F.interpolate(gaussian_feats.permute(0, 3, 1, 2), size=mujoco_feats_target.shape[1:3], mode="bilinear", align_corners=False)
            # features_final = torch.from_numpy(features_final).cuda().permute(0, 2, 3, 1)
            features_final = features_final.permute(0, 2, 3, 1)
            # print(features_final.shape, 'features_final')
            
            # compute loss between masked_gaussian_feats and features_final
            loss_feat = cosine_loss(features_final, mujoco_feats_target)
        
        if vis_features:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/images_{i}.png')
                
                image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/image_render_{i}.png')
            
            
        
        
        threshold = 0.5
        gaussian_masks_binary = (gaussian_masks > threshold).float()  # Convert to 0s and 1s
        ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
        gaussian_masks_scaled = (gaussian_masks_binary * 255).byte()  # uint8 format
        ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 4: Save masks as images
        os.makedirs('masks_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            # Save Gaussian mask
            gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
            # print(f"Saved {gaussian_filename}")

            # Save Ground Truth mask
            gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
            cv2.imwrite(gt_filename, ground_truth_masks_np[i])
            # print(f"Saved {gt_filename}")
        
        gaussian_depths_np = gaussian_depths.detach().cpu().numpy()
        mujoco_depths_np = mujoco_depths_target.detach().cpu().numpy()
        gaussian_depths_np *= 100
        mujoco_depths_np *= 100
        os.makedirs('depths_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            # Save Gaussian mask
            gaussian_filename = f"depths_inspection/gaussian_depth_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_depths_np[i])
            # print(f"Saved {gaussian_filename}")

            # Save Ground Truth mask
            gt_filename = f"depths_inspection/ground_truth_depth_{i}.png"
            cv2.imwrite(gt_filename, mujoco_depths_np[i])
            # print(f"Saved {gt_filename}")
            
            gt_filename = f"depths_inspection/compare_depth_{i}.png"
            cv2.imwrite(gt_filename, np.abs(mujoco_depths_np[i] - gaussian_depths_np[i]))
        
        # print(gaussian_depths.max(), gaussian_depths.min(), mujoco_depths_target.max(), mujoco_depths_target.min())
        # print(gaussian_depths.shape, 'gaussian_depths')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(mujoco_images_target.shape, 'mujoco_images_target')
        # print(mujoco_mask_target.shape, 'mujoco_mask_target')
        # print(gaussian_tensors.max(), mujoco_images_target.max())
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        # IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
        # print(IoU_loss, 'IoU_loss')
        IoU_loss_motion = iou_loss(gaussian_masks[:-1], motion_depth_near_mask)
        
        
                
        
        
        depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
        print(depth_loss.shape, 'depth_loss shape', depth_near_mask1.shape)
        depth_diff = (depth_loss * depth_near_mask1).mean()
        # print(depth_diff, 'depth_diff')
        # if use_feature_loss:
        #     print(l2_diff.item(), 'l2_diff', IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff', loss_feat.item(), 'loss_feat')
        # else:
        #     print(l2_diff.item(), 'l2_diff', IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff')
        
        # total_loss = l2_diff.sum()
        # total_loss = loss_feat + IoU_loss
        # total_loss = l2_diff.sum() + loss_feat + depth_diff * 1 + IoU_loss * 10
        flow_loss = flow_loss * 100
        flow_loss_motion = flow_loss_motion * 10
        # total_loss = flow_loss + flow_loss_motion
        # total_loss = depth_diff * 0.01 + flow_loss
        total_loss = IoU_loss_motion + flow_loss + depth_diff
        print(flow_loss.item(), 'flow_loss', flow_loss_motion.item(), 'flow_loss_motion', IoU_loss_motion.item(), 'IoU_loss_motion')
        print(depth_diff.item(), 'depth_diff')
        print(total_loss.item(), 'total_loss')
        # if IoU_loss > 0.85: 
        #     total_loss = l2_diff * 3 + depth_diff * 0.1 + IoU_loss * 10
        # else:
        #     total_loss = l2_diff + depth_diff * 1 + IoU_loss * 10
        # total_loss = 10 * depth_diff + 0.1 * IoU_loss
        total_loss.backward()

        all_losses = l2_diff.detach().cpu().numpy().tolist()
        all_images = [torch.clamp(tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() for tensor in gaussian_tensors]

        for optimizer in optimizers:
            optimizer.step()

    print('Camera results robot to world transformation:')
    for i in range(len(all_cameras)):
        print(all_cameras[i].robot_to_world())
        print(all_cameras[i].get_camera_pose().tolist(), 'camera pose final')
    # print(best_camera.robot_to_world())
    valid_masks = False
    if flow_loss < 0.5:
        valid_masks = True
    # yield (best_image, *rounded_final_params)
    # print(all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy())
    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), valid_masks
    
    
def pose_to_extrinsic(pose):
    # Extract translation and rotation components
    t_world = np.array(pose[:3])  # [tx, ty, tz]
    r = np.array(pose[3:])       # [rx, ry, rz], rotation vector

    # Compute rotation matrix using Rodrigues formula
    theta = np.linalg.norm(r)
    if theta < 1e-6:  # If rotation is negligible
        R_cam_to_world = np.eye(3)
    else:
        n = r / theta  # Unit axis
        # Skew-symmetric matrix
        K = np.array([[0, -n[2], n[1]],
                      [n[2], 0, -n[0]],
                      [-n[1], n[0], 0]])
        # Rodrigues formula
        I = np.eye(3)
        R_cam_to_world = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # World to camera rotation and translation
    R_world_to_cam = R_cam_to_world.T
    t_cam = -R_world_to_cam @ t_world

    # Construct 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_world_to_cam
    extrinsic[:3, 3] = t_cam

    return extrinsic

if __name__ == "__main__":
    Display = 0
    BATCH_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/droid1/2023-07-07'
    TRAJ_PATHs = [d for d in os.listdir(BATCH_PATH) if os.path.isdir(os.path.join(BATCH_PATH, d))]
    TRAJ_PATHs = sorted(TRAJ_PATHs)
    for i, traj_name in enumerate(TRAJ_PATHs):
        # if traj_name != 'Fri_Jul__7_09:50:13_2023':
        #     continue
        print(i, traj_name)
        TRAJ_PATH = os.path.join(BATCH_PATH, traj_name)
        results_file = os.path.join(TRAJ_PATH, 'results.pkl')
        results_dict = {}
        camera_names = [d for d in os.listdir(TRAJ_PATH) if os.path.isdir(os.path.join(TRAJ_PATH, d))]
        camera_names = sorted(camera_names)
        for j, cam_name in enumerate(camera_names):
            print(j, cam_name)
            if cam_name.split('_')[1] != 'left':
                continue
            # if cam_name != '24400334_left':
            #     continue
            DATA_PATH = os.path.join(TRAJ_PATH, cam_name)
            FRANKA = True
            UR5 = False
            ROBOTIQ = True
            cam_id = DATA_PATH.split('/')[-1]
            vis_features = True
            tracking_loss = True
            image_list = []
            depth_list = []
            mask_list = []
            grippers = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
            joints = np.load(os.path.join(DATA_PATH, 'joints.npy'))
            if UR5 or ROBOTIQ:
                grippers = grippers * 0
                fingers = np.stack([grippers, grippers, grippers, grippers, grippers, grippers, grippers, grippers], axis=1) # for ur5
            elif FRANKA:
                grippers = grippers * 0
                fingers = np.stack([grippers, grippers], axis=1) # for franka
            if UR5:
                offset_angle = 90 * np.pi / 180
                joints[:, 0] -= offset_angle
            joints = np.concatenate([joints, fingers], axis=1)
            print('joints: ', joints.shape)
            # print(joints)
            image_names = os.listdir(os.path.join(DATA_PATH, 'images'))
            image_names = sorted(image_names)
            for image_name in image_names:
                # print('image_name', image_name)
                image_path = os.path.join(DATA_PATH, 'images', image_name)
                img = Image.open(image_path)
                img = np.array(img)
                image_list.append(img)
            image_list = np.stack(image_list)
            _, H, W, C = image_list.shape
            # print(image_list.shape, 'images')
                
            depth_names = os.listdir(os.path.join(DATA_PATH, 'depths_fs'))
            depth_names = sorted(depth_names)
            for depth_name in depth_names:
                # print('depth_name', depth_name)
                depth_path = os.path.join(DATA_PATH, 'depths_fs', depth_name)
                depth = np.load(depth_path)
                depth_list.append(depth)
            depth_list = np.stack(depth_list)
            
            mask_names = os.listdir(os.path.join(DATA_PATH, 'masks'))
            mask_names = sorted(mask_names, key=lambda x: int(x.split('.')[0]))
            for mask_name in mask_names:
                # print('mask_name', mask_name)
                mask_path = os.path.join(DATA_PATH, 'masks', mask_name)
                mask = Image.open(mask_path)
                mask = np.array(mask)
                mask_list.append(mask)
            mask_list = np.stack(mask_list)
            print(mask_list.shape, 'masks ensure not 3 channels')
            print(grippers.shape, 'grippers')
            print(joints.shape, 'joints')
            print(image_list.shape, 'images')
            print(depth_list.shape, 'depths')
            assert(depth_list.shape[0] == image_list.shape[0] == grippers.shape[0] == joints.shape[0] == mask_list.shape[0])
            print('frames: ', image_list.shape[0])
            
            Current_Cams = {'19824535_left': np.array([-0.07528311, 0.02872137, 0.02331657, -0.32557736, 0.00624743, -1.56287453]),
                            '19824535_right': np.array([-0.06961441, -0.0353966, 0.02041079, -0.31886274, 0.00782202, -1.55214306]),
                            '22008760_left': np.array([0.43518401, 0.37382269, 0.34167107, -2.02259688, -0.08240233, -2.40839443]),
                            '22008760_right': np.array([0.34059767, 0.29106813, 0.35430316, -2.02269023, -0.07155337, -2.40794443]),
                            '23404442_left': np.array([0.25955495, -0.50010183, 0.4368819, -2.27734563, 0.33618977, -0.5369007]),
                            '23404442_right': np.array([0.35203881, -0.55772972, 0.40182455, -2.27980636, 0.34639743, -0.54752158]),
                            '24400334_left': np.array([2.59675732e-01, -3.66262596e-01, 2.48493048e-01, -1.74211540e+00, -1.21274269e-03, -7.14986722e-01]),
                            '24400334_right': np.array([0.35026008, -0.44562786, 0.25002654, -1.74644264, -0.00601653, -0.71559287]),
                            '26405488_left': np.array([0.11133259, 0.82074817, 0.04550905, -1.64633975, -0.17857016, -2.93766368]),
                            '26405488_right': np.array([-0.00941784, 0.79726688, 0.06931512, -1.64730253, -0.18917295, -2.9422189]),
                            '29838012_left': np.array([0.4154884, 0.4775471, 0.41143223, -2.21604102, -0.02257695, -2.96372386]),
                            '29838012_right': np.array([0.29402237, 0.45672118, 0.41110094, -2.20799482, -0.0194012, -2.95977621])}
            
            Optimal_param = {
                '22008760_left': {'w': torch.tensor([-0.0,  0.0, -0.0]), 'v': torch.tensor([ 0., -0., -0.])}, 
            }
            
            assert cam_id in Current_Cams.keys(), f'cam_id {cam_id} not in Current_Cams'
            intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))[0]
            extrinsics = np.load(os.path.join(DATA_PATH, 'extrinsics.npy'))
            pose = np.array([0.43518401, 0.37382269, 0.34167107, np.pi / 2, 0, 0]) # initial pose
            Optimal_Camera_Param = {'w': torch.tensor([-1.6827,  0.0755, -2.6683]), 'v': torch.tensor([ 0.4040,  0.4732,  0.2717])}
            # Optimal_Camera_Param = None
            # print(extrinsics, 'original')
            # c2w = np.linalg.inv(extrinsics)
            # print(c2w)
            # c2w[0, 3] = 0.5
            # c2w[1, 3] = 0.3
            # c2w[2, 3] = 0.3
            # print(c2w)
            # extrinsics = np.linalg.inv(c2w)
            # extrinsics = c2w
            # extrinsics[1, 3] = 0.2
            # extrinsics = np.eye(4)
            # extrinsics = pose_to_extrinsic(pose)
            # extrinsics = np.linalg.inv(extrinsics)
            # extrinsics[1, 3] = 0.2
            # extrinsics[0, 3] = -0.2
            # extrinsics[2, 3] = 1
            # print(extrinsics)
            # print(intrinsics)
            Optimize_list = []
            l = len(image_list)
            # // 5 if feature is used
            step_length = l // 10
            step_length = 10
            l = 100
            for ii in range(0, l, step_length):
                Optimize_list.append(ii)
            # print(Optimize_list)
            image_list = image_list[Optimize_list]
            depth_list = depth_list[Optimize_list]
            mask_list = mask_list[Optimize_list]
            joints = joints[Optimize_list]
            grippers = grippers[Optimize_list]
            fingers = fingers[Optimize_list]
            print(image_list.shape, 'images')
            print(depth_list.shape, 'depths')
            print(mask_list.shape, 'masks')
            print(grippers.shape, 'grippers')
            print(joints.shape, 'joints')
            print(fingers.shape, 'fingers')
            
            optimize_camera = True
            optimize_joints = False
            camera_lr = 0.01
            joints_lr = 0.001
            optimization_steps = 80
            powerful_optimize_dropdown = "Disabled"
            noise_input = 0.01
            num_inits_input = 1
        
            # Optimize_list = [0]
            # python optimize_multiframe_droid.py --model_path output/franka_fr3_2f85_1
            params, valid_optimization = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        powerful_optimize_dropdown, noise_input, num_inits_input)
            print(params)
            results_dict[cam_name] = {'extrinsics': params.tolist(), 'valid_optimization': valid_optimization}
            with open(results_file, 'wb') as f:
                pickle.dump(results_dict, f)
            