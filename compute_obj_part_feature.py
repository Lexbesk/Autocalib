import os
# from utils_loc.general_utils import pytorch_gc
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm, trange
import cv2
from typing import Any, Dict, Generator,List
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob

# python compute_obj_part_feature.py -s data/franka_emika_panda
# python compute_obj_part_feature.py -s /data/user_data/wenhsuac/chenyuzhang/drrobot/data/franka_fr3_2f85_finetune

def resize_image(img, longest_edge):
    # resize to have the longest edge equal to longest_edge
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)

def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag
    
def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-L/14@336px",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

def main(args):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    yolo_iou = 0.9
    yolo_conf = 0.4

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)

    base_dir = args.source_path
    # image_dir = os.path.join(base_dir, 'images')
    # if not os.path.exists(image_dir):
    #     image_dir = os.path.join(base_dir, 'color')
    # assert os.path.isdir(image_dir), f"Image directory {image_dir} does not exist."
    # dinov2_feat_dir = os.path.join(base_dir, 'dinov2_vits14')
    # os.makedirs(dinov2_feat_dir, exist_ok=True)

    image_paths = []
    
    canonical_data_dirs = [d for d in glob.glob(os.path.join(base_dir, "canonical_sample_*")) if os.path.isdir(d)]
    data_dirs = [d for d in glob.glob(os.path.join(base_dir, "sample_*")) if os.path.isdir(d)]
    # sort them
    canonical_data_dirs = sorted(canonical_data_dirs, key=lambda x: int(x.split("_")[-1]))
    data_dirs = sorted(data_dirs, key=lambda x: int(x.split("_")[-1]))
    all_dirs = canonical_data_dirs + data_dirs
    for data_dir in all_dirs:
        images = glob.glob(os.path.join(data_dir, "image_*.jpg"))
        # sort
        images = sorted(images, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for idx in range(len(images)):
            image_path = os.path.join(data_dir, "image_{}.jpg".format(idx))
            dino_path = os.path.join(data_dir, "dino_{}.npy".format(idx))
            print(image_path, dino_path)

            image = Image.open(image_path)
            image = resize_image(image, args.dino_resolution)
            image = dino_transform(image)[:3].unsqueeze(0)
            image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
            image = image.cuda()
            with torch.no_grad():
                features = dinov2.forward_features(image)["x_norm_patchtokens"][0]
            features = features.cpu().numpy()
            features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            features_chw = features_hwc.transpose((2, 0, 1))
            np.save(dino_path, features_chw)
    
    del dinov2
    # pytorch_gc()


if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--part_batch_size", type=int, default=32, help="Part-level CLIP inference batch size")
    parser.add_argument("--part_resolution", type=int, default=224, help="Part-level CLIP input image resolution")
    parser.add_argument("--sam_size", type=int, default=1024, help="Longest edge for MobileSAMV2 segmentation")
    parser.add_argument("--obj_feat_res", type=int, default=100, help="Intermediate (for MAP) SAM-enhanced Object-level feature resolution")
    parser.add_argument("--part_feat_res", type=int, default=400, help="Intermediate (for MAP) SAM-enhanced Part-level feature resolution")
    parser.add_argument("--final_feat_res", type=int, default=64, help="Final hierarchical CLIP feature resolution")
    parser.add_argument("--dino_resolution", type=int, default=800, help="Longest edge for DINOv2 feature generation")
    parser.add_argument("--mobilesamv2_encoder_name", type=str, default="mobilesamv2_efficientvit_l2", help="MobileSAMV2 encoder name")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
