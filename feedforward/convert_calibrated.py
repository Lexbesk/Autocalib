import os
import shutil
from tqdm import tqdm

def extract_data(root_folders, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find the next available folder number
    existing_folders = [f for f in os.listdir(output_folder) 
                        if os.path.isdir(os.path.join(output_folder, f)) and f.isdigit()]
    if existing_folders:
        folder_counter = max(int(f) for f in existing_folders) + 1
    else:
        folder_counter = 0
    
    # Process each root folder
    for root_folder in root_folders:
        for scene_folder in tqdm(os.listdir(root_folder)):
            scene_path = os.path.join(root_folder, scene_folder)
            if os.path.isdir(scene_path) and scene_folder.startswith("scene"):
                image0_path = os.path.join(scene_path, "images0")
                extrinsics_path = os.path.join(scene_path, "extrinsics.npy")
                if os.path.isdir(image0_path) and os.path.isfile(extrinsics_path):
                    for image_file in os.listdir(image0_path):
                        image_path = os.path.join(image0_path, image_file)
                        if os.path.isfile(image_path):
                            # Create new folder in output
                            new_folder = os.path.join(output_folder, str(folder_counter))
                            os.makedirs(new_folder, exist_ok=True)
                            # Copy image
                            shutil.copy(image_path, os.path.join(new_folder, image_file))
                            # Copy extrinsics and rename
                            shutil.copy(extrinsics_path, os.path.join(new_folder, "extrinsics.npy"))
                            folder_counter += 1

# Example usage
if __name__ == "__main__":
    root_folders = ["/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_2", "/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_3"]  # Replace with your root folder paths
    output_folder = "/data/group_data/katefgroup/datasets/chenyu/image_cam_pairs"  # Replace with your output folder path
    extract_data(root_folders, output_folder)