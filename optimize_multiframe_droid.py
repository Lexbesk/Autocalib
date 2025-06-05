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
python optimize_multiframe_droid.py --model_path output/franka_fr3_2f85_highres_finetune_0
This file is for UR5
"""


INSTRUCTIONS = """
# Optimization Workflow

## 1. Simple Joint Optimization 
To test a simple joint optimization:
1. Click "Random Initialize Joints" for MuJoCo to set random joint angles.
2. Click "Copy MuJoCo Joints" to transfer these parameters to the Gaussian renderer.
3. Slightly adjust one of the Gaussian joint angles (e.g., change θ1 by 0.1).
4. Check "Optimize Joints" and uncheck "Optimize Camera Parameters".
5. Click "Optimize" to start the optimization process.

## 2. Camera Optimization
To test camera parameter optimization:
1. Click "Random Initialize Camera" for MuJoCo to set random camera parameters.
2. Ensure the joint angles are the same for both MuJoCo and Gaussian renderer.
3. Check "Optimize Camera Parameters" and uncheck "Optimize Joints".
4. Click "Optimize" to start the optimization process.

## 3. Joint and Camera Optimization
To test both joint and camera optimization:
1. Use "Random Initialize Joints" and "Random Initialize Camera" for MuJoCo.
2. Check both "Optimize Joints" and "Optimize Camera Parameters".
3. Set appropriate learning rates and optimization steps.
4. Click "Optimize" to start the combined optimization process.
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
from gaussian_renderer import render, render_gradio
from scene.cameras import Camera_Pose, Camera
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss



def rotation_matrix_to_quaternion(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    return R.from_quat(quat).as_matrix()

gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()

use_feature_loss = True
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


def visualize_features(features, path):
    """
    Visualize a DINOv2 feature map by reducing its dimensionality to 3 using PCA.

    Args:
        features (np.ndarray): Feature map of shape (H, W, D), where D is the feature dimension (>700).

    Notes:
        - If your feature map is a PyTorch tensor, convert it to NumPy with features.cpu().numpy().
        - If it's a flat array of shape (N, D), reshape it to (H, W, D) first, where N = H * W.
    """
    # Get the dimensions of the feature map
    H, W, D = features.shape
    print(f"Feature map shape: {H}x{W}x{D}")

    # Reshape to (H*W, D) for PCA
    features_reshaped = features.reshape(H * W, D)

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)
    print(f"Explained variance ratio of 3 components: {pca.explained_variance_ratio_}")

    # Reshape back to (H, W, 3) for visualization
    features_pca_image = features_pca.reshape(H, W, 3)

    # Normalize each channel to [0,1] for RGB display
    for i in range(3):
        channel = features_pca_image[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:  # Avoid division by zero
            features_pca_image[:, :, i] = (channel - min_val) / (max_val - min_val)
        else:
            features_pca_image[:, :, i] = 0.5  # Set to mid-range if constant

    # Display the visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(features_pca_image)
    plt.axis('off')
    plt.title("DINOv2 Feature Visualization (PCA to RGB)")
    plt.savefig(path, bbox_inches='tight')

def render_scene(c_idx, *args):
    pixels = image_list[c_idx]
    return pixels

def reset_params():
    return [0] * n_joints + [0, -45, 2]

class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*args):
    n_params = len(args)
    n_joints = n_params - 3
    
    joint_angles = torch.tensor(args[:n_joints])
    azimuth, elevation, distance = args[n_joints:]

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
    joint_angles = torch.tensor(joints[Display])
    intrinsic_rh = intrinsics
    fx = intrinsic_rh[0, 0]
    fy = intrinsic_rh[1, 1]
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))
    camera_extrinsic_matrix = extrinsics
    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                            W, H, joint_pose=joint_angles, zero_init=True).cuda()
    if FRANKA:
        # frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
        frame = torch.clamp(render_gradio(example_camera_mujoco, gaussians, background_color, render_features = True)['render'], 0, 1)
    elif UR5:
        frame = torch.clamp(render_gradio(example_camera_mujoco, gaussians, background_color, render_features = True)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def gaussian_reset_params():
    return [0] * n_joints + [0, -45, 2]

def random_initialize_joints():
    return [round(np.random.uniform(-1, 1), 2) for _ in range(n_joints)]

def random_initialize_camera():
    random_azimuth = round(np.random.uniform(0, 360), 2)
    random_elevation = round(np.random.uniform(-90, 90), 2)
    random_distance = round(np.random.uniform(1, 3), 2)
    return [random_azimuth, random_elevation, random_distance]

def random_initialize():
    return random_initialize_joints() + random_initialize_camera()

def gaussian_random_initialize_joints():
    return random_initialize_joints()

def gaussian_random_initialize_camera():
    return random_initialize_camera()

def copy_mujoco_camera(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[n_joints:]

def copy_mujoco_joints(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[:n_joints]

def initial_render():
    initial_params = reset_params()
    mujoco_image = render_scene(Display, *initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    return mujoco_image, gaussian_image

def iou_loss(pred, target, smooth=1e-6):
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)  # (batch_size, H*W)
    target = target.view(batch_size, -1)
    intersection = (pred * target).sum(dim=1)  # Sum over pixels per batch
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou).mean()  # Average loss over batch

def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, powerful_optimize_dropdown, noise_input, num_inits_input, *all_params):
    n_params = len(all_params) // 2
    mujoco_params = all_params[:n_params]
    gaussian_params = list(all_params[n_params:])

    n_params = len(gaussian_params)
    n_joints = n_params - 3

    if powerful_optimize_dropdown == "Enabled":
        num_inits = int(num_inits_input)
        grid_size = int(np.sqrt(num_inits))
    else:
        num_inits = 1
        grid_size = 1
        noise_input = 0

    all_cameras = []
    all_joint_poses = []

    for init in range(num_inits):
        mujoco_masks = []
        mujoco_images = []
        mujoco_depths = []
        mujoco_feats = []
        mujoco_direct_feats = []
        first_camera = None  # Track the first camera to share its parameters
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Loading DINOv2 model...")
        os.makedirs('inspection_features', exist_ok=True)
        if use_feature_loss:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            dinov2 = dinov2.to(device)
            dino_transform = T.Compose([
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.5], std=[0.5]),
                                        ])
        for c_idx in range(image_list.shape[0]):
            # if c_idx != 3:
            #     continue
            # if c_idx not in Optimize_list:
            #     continue
            # print('c_idx', c_idx)
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
            
            mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1) # TODO permute?
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
            perturbed_params = gaussian_params.copy()
            
            if optimize_joints:
                perturbed_params[:n_joints] = [float(p) + np.random.normal(0, noise_input) for p in perturbed_params[:n_joints]]
            
            joint_angles = torch.tensor(perturbed_params[:n_joints], dtype=torch.float32, requires_grad=optimize_joints)
            
            if optimize_camera:
                perturbed_params[n_joints:] = [float(p) + np.random.normal(0, noise_input) for p in perturbed_params[n_joints:]]
            
            azimuth, elevation, distance = perturbed_params[n_joints:]

            dummy_cam = DummyCam(azimuth, elevation, distance)
            
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
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            # if i not in Optimize_list:
            #     print(i)
            #     i += 1
            #     continue
            camera.joint_pose = joint_pose
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features = use_feature_loss)
            elif UR5:
                output = render_gradio(camera, gaussians, background_color, render_features = use_feature_loss)
            
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
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
        if use_feature_loss:
            gaussian_feats = torch.stack(gaussian_feats)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        mujoco_mask_target = torch.stack(mujoco_masks)
        mujoco_images_target = torch.stack(mujoco_images)
        mujoco_depths_target = torch.stack(mujoco_depths)
        
        if use_feature_loss:
            mujoco_feats_target = torch.stack(mujoco_feats)
            mujoco_direct_feats_target = torch.stack(mujoco_direct_feats)
            features_final = F.interpolate(gaussian_feats.permute(0, 3, 1, 2), size=mujoco_feats_target.shape[1:3], mode="bilinear", align_corners=False)
            # features_final = torch.from_numpy(features_final).cuda().permute(0, 2, 3, 1)
            features_final = features_final.permute(0, 2, 3, 1)
            # print(features_final.shape, 'features_final')
            
            # compute loss between masked_gaussian_feats and features_final
            loss_feat = cosine_loss(features_final, mujoco_feats_target)
            
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("Loading DINOv2 model...")
            image = gaussian_tensors
            mujoco_images_target1 = mujoco_images_target / 255
            
            # print(image.shape, 'image_shape')
            resized_image = resize_tensor(image, longest_edge=800)
            resized_mujo_image = resize_tensor(mujoco_images_target1, longest_edge=800)
            # print(resized_image.shape, 'resized_image_shape')
            normalize = T.Normalize(mean=[0.5], std=[0.5])
            
            image = (resized_image[0] * 255).permute(1, 2, 0)
            image_mujo = (resized_mujo_image[0] * 255).permute(1, 2, 0)
            image_np = image.to(torch.uint8)
            image_np_mujo = image_mujo.to(torch.uint8)
            image_np = image_np.detach().cpu().numpy()
            image_np_mujo = image_np_mujo.detach().cpu().numpy()
            image_np = Image.fromarray(image_np)
            image_np_mujo = Image.fromarray(image_np_mujo)
            image_np.save(f'inspection_features/image_final_{0}.png')
            image_np_mujo.save(f'inspection_features/image_mujo_{0}.png')
            
            image_final = normalize(resized_image)
            mujo_final = normalize(resized_mujo_image)
            dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
            dino_mojo, _, _ = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
            features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
            mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
            print(image_final.shape, 'image_shape_final')
            print(features.shape, 'dino image features render')
            print(mujo_features.shape, 'mujoco image features render')
            print(mujoco_direct_feats_target.shape, 'mujoco_direct_feats_target')
            
            loss_direct_feat = cosine_loss(features, mujo_features)
            
            if vis_features:
                features1 = features[0].detach().cpu().numpy()
                features_render = features1.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
                
                features2 = mujoco_direct_feats_target[0].detach().cpu().numpy()
                features_hwc = features2.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
                
                resized_mask = F.interpolate(gaussian_masks[0].unsqueeze(0).unsqueeze(0).float(), size=(features_hwc.shape[:2]), mode='nearest')
                masked_features_hwc = features_hwc * resized_mask.squeeze().unsqueeze(-1).detach().cpu().numpy()
                masked_features_render = features_render * resized_mask.squeeze().unsqueeze(-1).detach().cpu().numpy()
                # features_final = F.interpolate(masked_gaussian_feats.unsqueeze(0).permute(0, 3, 1, 2), size=features_hwc.shape[:2], mode="bilinear", align_corners=False).detach().cpu().numpy()
                # features_final = features_final.squeeze().transpose(1, 2, 0)
                print(features_hwc.shape, 'features_hwc')
                print(features_render.shape, 'features_final')
                print(resized_mask.shape, 'resize mask')
                features_final = np.concatenate([features_hwc, features_render], axis=0)
                features_final_masked = np.concatenate([masked_features_hwc, masked_features_render], axis=0)
                visualize_features(features_hwc, f'inspection_features/dino_direct_features_{0}.png')
                visualize_features(features_render, f'inspection_features/dino_direct_features_render{0}.png')
                visualize_features(masked_features_hwc, f'inspection_features/dino_direct_features_masked{0}.png')
                visualize_features(masked_features_render, f'inspection_features/dino_direct_features_masked_render{0}.png')
                visualize_features(features_final, f'inspection_features/dino_direct_features_final{0}.png')
                visualize_features(features_final_masked, f'inspection_features/dino_direct_features_final_masked{0}.png')
        
        
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
        gaussian_depths_np *= 40
        mujoco_depths_np *= 40
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
        
        # print(gaussian_depths.max(), gaussian_depths.min(), mujoco_depths_target.max(), mujoco_depths_target.min())
        # print(gaussian_depths.shape, 'gaussian_depths')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(mujoco_images_target.shape, 'mujoco_images_target')
        # print(mujoco_mask_target.shape, 'mujoco_mask_target')
        # print(gaussian_tensors.max(), mujoco_images_target.max())
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
        # print(IoU_loss, 'IoU_loss')
        
        
        depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
        depth_diff = depth_loss.mean()
        # print(depth_diff, 'depth_diff')
        if use_feature_loss:
            print(l2_diff.item(), 'l2_diff', IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff', loss_feat.item(), 'loss_feat')
        else:
            print(l2_diff.item(), 'l2_diff', IoU_loss.item(), 'IoU_loss', depth_diff.item(), 'depth_diff')
        
        # total_loss = l2_diff.sum()
        # total_loss = loss_feat + IoU_loss
        # total_loss = l2_diff.sum() + loss_feat + depth_diff * 1 + IoU_loss * 10
        if IoU_loss > 0.7: 
            total_loss = l2_diff * 3 + depth_diff * 0.1 + loss_feat * 10
        else:
            total_loss = l2_diff + depth_diff * 1 + loss_feat * 1 + IoU_loss * 10
        # total_loss = 10 * depth_diff + 0.1 * IoU_loss
        total_loss.backward()

        all_losses = l2_diff.detach().cpu().numpy().tolist()
        all_images = [torch.clamp(tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() for tensor in gaussian_tensors]

        for optimizer in optimizers:
            optimizer.step()

        if step % 10 == 0 or step == optimization_steps - 1:
            grid_images = np.zeros((H * grid_size, W * grid_size, 3))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < len(all_images):
                        grid_images[i*H:(i+1)*H, j*W:(j+1)*W] = all_images[idx]

            best_idx = np.argmin(all_losses)
            best_camera = all_cameras[best_idx]
            best_joint_pose = all_joint_poses[best_idx]
            
            # Extract updated parameters for the best camera
            updated_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
            updated_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
            updated_camera_params = extract_camera_parameters(updated_extrinsic.T)
            
            updated_params = updated_joint_angles + [
                updated_camera_params['azimuth'],
                updated_camera_params['elevation'],
                updated_camera_params['distance']
            ]

            rounded_params = [round(param, 2) for param in updated_params]

            yield (grid_images, *rounded_params)

    # Final yield with only the best result
    best_idx = np.argmin(all_losses)
    best_image = all_images[best_idx]
    best_camera = all_cameras[best_idx]
    best_joint_pose = all_joint_poses[best_idx]

    # Extract final parameters for the best camera
    final_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
    final_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
    final_camera_params = extract_camera_parameters(final_extrinsic.T)
    print('Camera results robot to world transformation:')
    for i in range(len(all_cameras)):
        print(all_cameras[i].robot_to_world())
        print(all_cameras[i].get_camera_pose().tolist(), 'camera pose final')
    # print(best_camera.robot_to_world())
    
    final_params = final_joint_angles + [
        final_camera_params['azimuth'],
        final_camera_params['elevation'],
        final_camera_params['distance']
    ]

    rounded_final_params = [round(param, 2) for param in final_params]

    # yield (best_image, *rounded_final_params)
    yield (all_images[Display], *rounded_final_params)

with gr.Blocks() as demo:
    
    with gr.Row():
        mujoco_output_image = gr.Image(type="numpy", label="MuJoCo Rendered Scene")
        gaussian_output_image = gr.Image(type="numpy", label="Gaussian Rendered Scene")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## MuJoCo Renderer")
            with gr.Row():
                mujoco_joint_inputs = [gr.Number(label=f"θ{i+1}", value=0) for i in range(n_joints)]
            
            with gr.Row():
                mujoco_camera_inputs = [
                    gr.Number(label="Azimuth (deg)", value=0),
                    gr.Number(label="Elevation (deg)", value=-45),
                    gr.Number(label="Distance (m)", value=2),
                ]
            
            with gr.Row():
                mujoco_render_button = gr.Button("Render MuJoCo", scale=1)
                mujoco_reset_button = gr.Button("Reset MuJoCo", scale=1)
                mujoco_random_joints_button = gr.Button("Random Initialize Joints", scale=1)
                mujoco_random_camera_button = gr.Button("Random Initialize Camera", scale=1)
            
            mujoco_all_inputs = mujoco_joint_inputs + mujoco_camera_inputs
            mujoco_render_button.click(fn=render_scene, inputs=mujoco_all_inputs, outputs=mujoco_output_image)
            mujoco_reset_button.click(fn=reset_params, outputs=mujoco_all_inputs)
            mujoco_random_joints_button.click(fn=random_initialize_joints, outputs=mujoco_joint_inputs)
            mujoco_random_camera_button.click(fn=random_initialize_camera, outputs=mujoco_camera_inputs)
        
        with gr.Column():
            gr.Markdown("## Gaussian Renderer")
            with gr.Row():
                gaussian_joint_inputs = [gr.Number(label=f"θ{i+1}", value=0) for i in range(n_joints)]
            
            with gr.Row():
                gaussian_camera_inputs = [
                    gr.Number(label="Azimuth (deg)", value=0),
                    gr.Number(label="Elevation (deg)", value=-45),
                    gr.Number(label="Distance (m)", value=2),
                ]
            
            with gr.Row():
                gaussian_render_button = gr.Button("Render Gaussian", scale=1)
                gaussian_reset_button = gr.Button("Reset Gaussian", scale=1)
                gaussian_random_joints_button = gr.Button("Random Initialize Joints", scale=1)
                gaussian_random_camera_button = gr.Button("Random Initialize Camera", scale=1)
            
            with gr.Row():
                gaussian_copy_camera_button = gr.Button("Copy MuJoCo Camera", scale=1)
                gaussian_copy_joints_button = gr.Button("Copy MuJoCo Joints", scale=1)
            
            with gr.Row():
                with gr.Column(scale=1):
                    optimize_camera = gr.Checkbox(label="Optimize Camera Parameters", value=False)
                    optimize_joints = gr.Checkbox(label="Optimize Joints", value=False)
                with gr.Column(scale=1):
                    camera_lr = gr.Number(label="Camera Learning Rate", value=0.02)
                    joints_lr = gr.Number(label="Joints Learning Rate", value=0.02)
                with gr.Column(scale=1):
                    optimization_steps = gr.Number(label="Optimization Steps", value=50, step=1)
                    optimize_button = gr.Button("Optimize", scale=1)
            
            with gr.Row():
                powerful_optimize_dropdown = gr.Dropdown(
                    label="Powerful Optimize",
                    choices=["Disabled", "Enabled"],
                    value="Disabled"
                )
            
            with gr.Row():
                noise_input = gr.Number(
                    label="Insert Noise Amount",
                    value=0.01,
                    step=0.001,
                    precision=3,
                    visible=False
                )
                num_inits_input = gr.Dropdown(
                    label="Number of Initializations",
                    choices=["1", "4", "9", "16"],
                    value="1",
                    visible=False
                )
            
            gaussian_all_inputs = gaussian_joint_inputs + gaussian_camera_inputs
            gaussian_render_button.click(fn=gaussian_render_scene, inputs=gaussian_all_inputs, outputs=gaussian_output_image)
            gaussian_reset_button.click(fn=gaussian_reset_params, outputs=gaussian_all_inputs)
            gaussian_random_joints_button.click(fn=gaussian_random_initialize_joints, outputs=gaussian_joint_inputs)
            gaussian_random_camera_button.click(fn=gaussian_random_initialize_camera, outputs=gaussian_camera_inputs)
            
            gaussian_copy_camera_button.click(
                fn=copy_mujoco_camera,
                inputs=mujoco_all_inputs,
                outputs=gaussian_camera_inputs
            )
            
            gaussian_copy_joints_button.click(
                fn=copy_mujoco_joints,
                inputs=mujoco_all_inputs,
                outputs=gaussian_joint_inputs
            )
            
            optimize_button.click(
                fn=optimize,
                inputs=[optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        powerful_optimize_dropdown, noise_input, num_inits_input] + 
                    mujoco_all_inputs + gaussian_all_inputs,
                outputs=[gaussian_output_image] + gaussian_all_inputs,
                show_progress=True
            )
        
    gr.Markdown("# Robot Scene Renderer Comparison")
    gr.Markdown(INSTRUCTIONS)   
    
    demo.load(fn=initial_render, outputs=[mujoco_output_image, gaussian_output_image])

    # Move this inside the gr.Blocks() context
    powerful_optimize_dropdown.change(
        fn=lambda x: [gr.update(visible=(x == "Enabled"))] * 3,
        inputs=[powerful_optimize_dropdown],
        outputs=[noise_input, num_inits_input]
    )
    
    
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
    DATA_PATH = '/data/user_data/wenhsuac/chenyuzhang/rh20t_output/rh20t_gs_splatting_logs/drrobot_data'
    # DATA_PATH = '/data/user_data/wenhsuac/chenyuzhang/rh20t_output/rh20t_gs_splatting_logs/drrobot_data_f0172289_task_0101_user_0007_scene_0009_cfg_0004'
    # DATA_PATH = '/data/user_data/wenhsuac/chenyuzhang/rh20t_output/rh20t_gs_splatting_logs/drrobot_data_f0172289_task_0101_user_0015_scene_0002_cfg_0004'
    DATA_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/droid1/2023-07-07/Fri_Jul__7_09:50:13_2023/22008760_left'
    # DATA_PATH = '/data/user_data/wenhsuac/chenyuzhang/drrobot/rh20t_data/f0461559_task_0122_user_0007_scene_0002_cfg_0005'
    FRANKA = True
    UR5 = False
    ROBOTIQ = True
    cam_id = DATA_PATH.split('/')[-1]
    vis_features = True
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
    print(joints)
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
    print(image_list.shape, 'images')
        
    depth_names = os.listdir(os.path.join(DATA_PATH, 'depths'))
    depth_names = sorted(depth_names)
    for depth_name in depth_names:
        # print('depth_name', depth_name)
        depth_path = os.path.join(DATA_PATH, 'depths', depth_name)
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
    # Optimal_Camera_Param = {'w': torch.tensor([-1.6827,  0.0755, -2.6683]), 'v': torch.tensor([ 0.4040,  0.4732,  0.2717])}
    Optimal_Camera_Param = None
    print(extrinsics, 'original')
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
    step_length = l // 2
    for ii in range(0, l, step_length):
        Optimize_list.append(ii)
    print(Optimize_list)
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
    # Optimize_list = [0]
    # python optimize_multiframe_droid.py --model_path output/franka_fr3_2f85_1
    
    
    
    
    demo.launch(share=True, server_port=8080)
    