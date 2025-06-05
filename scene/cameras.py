#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils_loc.graphics_utils import getWorld2View2, getProjectionMatrix, se3_to_SE3
from scipy.spatial.transform import Rotation

def extrinsic_to_euler(extrinsic, euler_order='xyz'):
    """
    Convert a 4x4 world-to-camera extrinsic matrix back to a 6D camera pose.

    Args:
        extrinsic (np.ndarray): 4x4 extrinsic matrix.
        euler_order (str): Order of Euler angles (e.g., 'xyz'). Default is 'xyz'.

    Returns:
        np.ndarray: 6-element array [x, y, z, alpha, beta, gamma].
    """
    # Extract rotation and translation from extrinsic matrix
    R_wc = extrinsic[:3, :3]  # World-to-camera rotation
    t = extrinsic[:3, 3]      # Translation vector

    # Compute camera position C
    C = -R_wc.T @ t  # C = -R_wc^T * t

    # Compute camera-to-world rotation
    R_cw = R_wc.T

    # Convert rotation matrix to Euler angles
    euler_angles = Rotation.from_matrix(R_cw).as_euler(euler_order)

    # Combine into 6D pose
    pose_6d = np.concatenate([C, euler_angles])
    return pose_6d

class Camera_Pose(nn.Module):
    def __init__(self,start_pose_w2c, FoVx, FoVy, image_width, image_height, joint_pose=None, time = 0, depth=None,
             zero_init=False):
        super(Camera_Pose, self).__init__()
        FoVx = torch.tensor(FoVx, dtype=torch.float32) if not isinstance(FoVx, torch.Tensor) else FoVx
        FoVy = torch.tensor(FoVy, dtype=torch.float32) if not isinstance(FoVy, torch.Tensor) else FoVy
        tanfovx = torch.tan(FoVx * 0.5)
        tanfovy = torch.tan(FoVy * 0.5)
        fx = image_width / (2 * tanfovx)
        fy = image_height / (2 * tanfovy)
        self.fx = nn.Parameter(fx, requires_grad=True)
        self.fy = fx
        self.FoVx = 2 * torch.atan(image_width / (2 * self.fx))
        self.FoVy = 2 * torch.atan(image_height / (2 * self.fy))

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.cov_offset = 0
        self.time = time

        self.depth = depth
        
    
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        # self.w = nn.Parameter(torch.tensor([0.2, 0.2, 0.2]).to(start_pose_w2c.device))
        # self.v = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]).to(start_pose_w2c.device))
        self.device = start_pose_w2c.device
        self.start_pose_w2c = start_pose_w2c
         
        # deltaT=se3_to_SE3(self.w,self.v).to(start_pose_w2c.device)
        # self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.detach().inverse()).inverse()
        # self.world_view_transform = self.pose_w2c.transpose(0, 1).requires_grad_(True)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.device)
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        
        self.joint_pose = joint_pose

    @property
    def world_view_transform(self):
        return torch.matmul(se3_to_SE3(self.w, self.v), self.start_pose_w2c.detach().inverse()).inverse().transpose(0, 1)
    
    def robot_to_world(self):
        print('v', self.v)
        print('w', self.w)
        return se3_to_SE3(self.w, self.v).clone().detach().inverse()
    
    def get_params(self):
        return self.w, self.v
    
    @torch.no_grad()
    def tcp_translation(self, tcp, gt_extrinsics=None):
        R2W = self.robot_to_world()
        if gt_extrinsics is not None:
            gt_extrinsics = torch.tensor(gt_extrinsics, dtype=torch.float32).cuda()
            extrinsic = self.world_view_transform.transpose(0, 1)
            tcp_hom = np.append(tcp, 1)
            tcp_hom = torch.tensor(tcp_hom, dtype=torch.float32).cuda()
            tcp_hom1 = torch.matmul(extrinsic, tcp_hom)
            tcp_hom = torch.matmul(gt_extrinsics, tcp_hom)
            # print(extrinsic, gt_extrinsics)
            distance = torch.norm(tcp_hom[:3] - tcp_hom1[:3])
        else:
            tcp_hom = np.append(tcp, 1)
            tcp_hom = torch.tensor(tcp_hom, dtype=torch.float32).cuda()
            tcp_hom1 = torch.matmul(R2W, tcp_hom)
            distance = torch.norm(tcp_hom[:3] - tcp_hom1[:3])
        return distance
    
    def get_camera_pose(self):
        w2c = self.world_view_transform.transpose(0, 1)
        # print(extrinsic_to_euler(w2c.cpu().numpy(), euler_order='xyz'))
        return extrinsic_to_euler(w2c.detach().cpu().numpy(), euler_order='xyz')
        
    @property
    def full_proj_transform(self):
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.device)
        full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        return full_proj_transform
        

    @torch.no_grad()
    def transformation_distance(self, gt_extrinsics=None):
        """
        Calculate the rotational and translational distance between a given 4x4 transformation matrix T
        and the identity transformation matrix.

        Parameters:
        T (numpy.ndarray): A 4x4 transformation matrix with a 3x3 rotation matrix in the top-left
                        and a 3x1 translation vector in the top-right.

        Returns:
        tuple: (rotational_angle_deg, translational_distance)
            - rotational_angle_deg (float): The rotational difference in degrees.
            - translational_distance (float): The Euclidean distance of the translation vector.
        """
        # Extract the rotation matrix R (top-left 3x3) and translation vector t (top-right 3x1)
        T = se3_to_SE3(self.w, self.v).clone().detach().cpu().inverse()
        if gt_extrinsics is not None:
            gt_extrinsics = torch.tensor(gt_extrinsics, dtype=torch.float32).cuda()
            extrinsic = self.world_view_transform.transpose(0, 1).inverse()
            T = torch.matmul(extrinsic, gt_extrinsics).cpu()
            
        R = T[:3, :3]
        t = T[:3, 3]

        # Compute the trace of R
        trace_R = np.trace(R)

        # Compute the argument for arccos, ensuring it stays within [-1, 1] for numerical stability
        arg = (trace_R - 1) / 2
        arg = np.clip(arg, -1, 1)

        # Compute the rotation angle in radians
        theta_rad = np.arccos(arg)

        # Convert the rotation angle to degrees
        theta_deg = np.degrees(theta_rad)

        # Compute the translational distance as the Euclidean norm of t
        d = np.linalg.norm(t)

        return theta_deg, d

    # def forward(self, start_pose_w2c):
    #     deltaT=se3_to_SE3(self.w,self.v).to(start_pose_w2c.device)
    #     self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
    #     self.update()
    
    # def current_campose_c2w(self):
    #     return self.pose_w2c.inverse().clone().cpu().detach().numpy()

    # def update(self):
    #     self.world_view_transform = self.pose_w2c.transpose(0, 1).to(self.device).requires_grad_(True)
    #     # print("world view transform", self.world_view_transform)
    #     print("requires grad?", self.world_view_transform.requires_grad)
    #     self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.device)
    #     self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    #     self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cpu", joint_pose=None, depth=None, dino_feat_chw=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.dino_feat_chw = dino_feat_chw

        self.data_device = torch.device(data_device)
        self.original_image = torch.from_numpy(image).clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        

        self.joint_pose = torch.from_numpy(joint_pose).to(self.data_device)
        self.depth = torch.from_numpy(depth).to(self.data_device) if depth is not None else None
        self.robot_mask = self.depth < 10.0


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

