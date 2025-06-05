import numpy as np
import transformations
import torch

def generate_camera_poses(category, num_samples=50):
    # Define angle ranges (in degrees) and radius range (in meters)
    if category == 'left':
        theta_min, theta_max = 30, 50  # Elevation from z-axis
        phi_min, phi_max = 40, 70     # Left side, centered around 90°
    elif category == 'right':
        theta_min, theta_max = 20, 60
        phi_min, phi_max = -120, -60   # Right side, centered around -90°
    elif category == 'top':
        theta_min, theta_max = 5, 25   # Near top
        phi_min, phi_max = 0, 360      # All around
    else:
        raise ValueError("Invalid category. Use 'left', 'right', or 'top'.")

    # Randomly sample angles and radius
    np.random.seed(42)  # For reproducibility; remove for true randomness
    theta_deg = np.random.uniform(theta_min, theta_max, num_samples)
    phi_deg = np.random.uniform(phi_min, phi_max, num_samples)
    r = np.random.uniform(0.9, 1.5, num_samples)

    # Convert to radians for computation
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    # Generate positions in spherical coordinates
    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)
    positions = np.stack([x, y, z], axis=1)  # Shape: (num_samples, 3)

    # Define roll angles for orientation variation (in degrees)
    roll_degrees = [-15, 0, 15]
    roll_radians = np.deg2rad(roll_degrees)

    # Target point: workspace where arm and object are
    target = np.array([0.6, 0.0, 0.5])

    # Generate poses
    poses = []
    for pos in positions:
        # Compute base rotation matrix to look at the target
        R_base, angles = camera_extrinsic_and_pose(pos, target)
        # R_base = np.linalg.inv(R_base)
        # R_base = look_at(pos, target, up=np.array([0, 1, 0]))
        # for roll in roll_radians:
        #     # Add roll around camera's z-axis
        #     R_z = np.array([
        #         [np.cos(roll), -np.sin(roll), 0],
        #         [np.sin(roll), np.cos(roll), 0],
        #         [0, 0, 1]
        #     ])
        #     R_final = R_base @ R_z  # Post-multiply to apply roll in camera frame

            # Convert to XYZ Euler angles
            # roll_euler, pitch_euler, yaw_euler = matrix_to_euler_xyz(R_final)
        roll_euler, pitch_euler, yaw_euler = transformations.euler_from_matrix(R_base, 'sxyz')
        # Collect pose
        pose = [float(pos[0]), float(pos[1]), float(pos[2]),
                roll_euler, pitch_euler, yaw_euler]
        poses.append(pose)

    return poses


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def camera_extrinsic_and_pose(camera_pos, target_pos):
    """
    Compute the camera extrinsic matrix and Euler angles (XYZ order) for a camera
    at position `camera_pos` looking at `target_pos`, with up direction [0,1,0].

    Parameters:
    - camera_pos (list or np.array): 3D position of the camera in world coordinates.
    - target_pos (list or np.array): 3D position of the target in world coordinates.

    Returns:
    - M (np.array): 4x4 extrinsic matrix (world to camera coordinates).
    - euler_angles (np.array): Euler angles in degrees [theta_x, theta_y, theta_z].
    """
    # Convert inputs to NumPy arrays
    C = np.array(camera_pos)
    T = np.array(target_pos)

    # Compute the camera's Z-axis: direction from camera to target
    Z_cam = T - C
    Z_cam = Z_cam / np.linalg.norm(Z_cam)

    # Define the up direction as [0,1,0]
    U = np.array([0, 0, -1])

    # Compute the camera's X-axis: perpendicular to U and Z_cam
    X_cam = np.cross(U, Z_cam)
    X_cam = X_cam / np.linalg.norm(X_cam)

    # Compute the camera's Y-axis: to form a right-handed coordinate system
    Y_cam = np.cross(Z_cam, X_cam)

    # Form the rotation matrix R (columns are X_cam, Y_cam, Z_cam)
    R = np.column_stack((X_cam, Y_cam, Z_cam))

    # Compute the translation vector t = -R @ C
    t = -R @ C

    # Assemble the 4x4 extrinsic matrix M
    M = np.row_stack((np.column_stack((R, t)), [0, 0, 0, 1]))

    # Extract Euler angles in XYZ order from the rotation matrix
    theta_y = np.arcsin(R[0, 2])  # R[0,2] is R13
    theta_x = np.arctan2(-R[1, 2], R[2, 2])  # -R23, R33
    theta_z = np.arctan2(-R[0, 1], R[0, 0])  # -R12, R11

    # Convert Euler angles from radians to degrees
    euler_angles = np.degrees([theta_x, theta_y, theta_z])

    return M, euler_angles



def rot_x(theta):
    """Compute rotation matrix around the X-axis for given angles."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    return torch.stack([
        one, zero, zero,
        zero, c, -s,
        zero, s, c
    ], dim=-1).view(-1, 3, 3)

def rot_y(theta):
    """Compute rotation matrix around the Y-axis for given angles."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    return torch.stack([
        c, zero, s,
        zero, one, zero,
        -s, zero, c
    ], dim=-1).view(-1, 3, 3)

def rot_z(theta):
    """Compute rotation matrix around the Z-axis for given angles."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    return torch.stack([
        c, -s, zero,
        s, c, zero,
        zero, zero, one
    ], dim=-1).view(-1, 3, 3)

def perturb_extrinsic(E, thetas_x, thetas_y, thetas_z, dxs, dys, dzs):
    """
    Perturb an extrinsic matrix to generate nearby camera views.

    Args:
        E (torch.Tensor): Original extrinsic matrix, shape (4, 4).
        thetas_x (torch.Tensor): Rotation angles around X-axis (radians).
        thetas_y (torch.Tensor): Rotation angles around Y-axis (radians).
        thetas_z (torch.Tensor): Rotation angles around Z-axis (radians).
        dxs (torch.Tensor): Translation displacements along X-axis.
        dys (torch.Tensor): Translation displacements along Y-axis.
        dzs (torch.Tensor): Translation displacements along Z-axis.

    Returns:
        torch.Tensor: Perturbed extrinsic matrices, shape (n_perturb, 4, 4),
                      where n_perturb = len(thetas_x) * len(thetas_y) * len(thetas_z) * len(dxs) * len(dys) * len(dzs).
    """
    # Extract rotation and translation from the extrinsic matrix
    R = E[:3, :3]  # shape (3, 3)
    t = E[:3, 3:4]  # shape (3, 1)

    # Compute rotation matrices for all perturbation angles
    R_x = rot_x(thetas_x)  # shape (n_theta_x, 3, 3)
    R_y = rot_y(thetas_y)  # shape (n_theta_y, 3, 3)
    R_z = rot_z(thetas_z)  # shape (n_theta_z, 3, 3)

    # Compute all possible R_delta combinations using broadcasting
    # Resulting shape: (n_theta_z, n_theta_y, n_theta_x, 3, 3)
    R_delta = torch.matmul(R_z[:, None, None], torch.matmul(R_y[None, :, None], R_x[None, None, :]))
    n_rot = R_delta.shape[0] * R_delta.shape[1] * R_delta.shape[2]
    R_delta = R_delta.view(n_rot, 3, 3)  # shape (n_rot, 3, 3)

    # Compute all possible delta_t_c combinations
    all_delta_t_c = torch.cartesian_prod(dxs, dys, dzs)  # shape (n_trans, 3)
    n_trans = all_delta_t_c.shape[0]
    all_delta_t_c = all_delta_t_c.view(n_trans, 3, 1)  # shape (n_trans, 3, 1)

    # Compute R_delta^T for all rotations
    R_delta_T = R_delta.transpose(-2, -1)  # shape (n_rot, 3, 3)

    # Compute perturbed rotation: R' = R_delta^T @ R
    R_prime = torch.matmul(R_delta_T, R)  # shape (n_rot, 3, 3)

    # Compute intermediate term: R_delta^T @ t
    temp = torch.matmul(R_delta_T, t)  # shape (n_rot, 3, 1)

    # Compute R_delta^T @ delta_t_c for all combinations using broadcasting
    R_delta_T_delta_t_c = torch.matmul(R_delta_T[:, None], all_delta_t_c[None, :])  # shape (n_rot, n_trans, 3, 1)

    # Compute perturbed translation: t' = R_delta^T @ t - R_delta^T @ delta_t_c
    t_prime = temp[:, None] - R_delta_T_delta_t_c  # shape (n_rot, n_trans, 3, 1)

    # Expand R_prime to match t_prime's shape
    R_prime_exp = R_prime[:, None].expand(-1, n_trans, 3, 3)  # shape (n_rot, n_trans, 3, 3)

    # Construct the top 3x4 part of the extrinsic matrices
    E_top = torch.cat([R_prime_exp, t_prime], dim=3)  # shape (n_rot, n_trans, 3, 4)

    # Add the last row [0, 0, 0, 1]
    last_row = torch.tensor([[0, 0, 0, 1]], dtype=E.dtype).expand(n_rot, n_trans, 1, 4)
    E_prime = torch.cat([E_top, last_row], dim=2)  # shape (n_rot, n_trans, 4, 4)

    # Reshape to a single tensor of all perturbed extrinsics
    E_prime = E_prime.view(n_rot * n_trans, 4, 4)  # shape (n_perturb, 4, 4)

    return E_prime



# Example usage
if __name__ == "__main__":
    left_poses = generate_camera_poses('right')
    print(f"Generated {len(left_poses)} poses for 'left' category.")
    print("Sample pose:", left_poses[0])
    
    
