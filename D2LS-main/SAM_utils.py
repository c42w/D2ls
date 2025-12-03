import numpy as np
import cv2
import os
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries
import torch

# 初始化SAM模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")  # 加载SAM权重
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    crop_nms_thresh=0.5,
    box_nms_thresh=0.5,
    pred_iou_thresh=0.96  # 调整参数以适应遥感场景
)


def generate_sam_features(image_path, save_dir):
    """为LOVEDA图像生成SAM的对象掩码和边界先验"""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # SAM生成掩码
    masks = mask_generator.generate(image_np)
    if not masks:
        return

    # 生成对象掩码（前50个最大对象）
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    obj_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    for idx, mask in enumerate(sorted_masks[:50]):  # 取前50个最大对象
        obj_mask[mask['segmentation']] = idx + 1  # 对象ID从1开始

    # 生成边界先验
    boundary_prior = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    for mask in masks:
        seg = mask['segmentation'].astype(np.uint8)
        boundary = find_boundaries(seg, mode='thick')  # 提取边界
        boundary_prior[boundary] = 1  # 边界标记为1

    # 保存结果
    img_name = os.path.basename(image_path).split(".")[0]
    Image.fromarray(obj_mask).save(os.path.join(save_dir, f"{img_name}_obj.png"))
    Image.fromarray((boundary_prior * 255).astype(np.uint8)).save(os.path.join(save_dir, f"{img_name}_boundary.png"))


# 处理LOVEDA数据集
loveda_img_dir = r"E:\10-User\ChenWei\D2LS-SAM\D2LS-main\data\LoveDA\train_val\Urban\images_png"  # LOVEDA图像路径
sam_feature_dir = r"E:\10-User\ChenWei\D2LS-SAM\D2LS-main\data\LoveDA\train_val\Urban\SAM"  # 保存SAM特征的路径
os.makedirs(sam_feature_dir, exist_ok=True)

for img_file in os.listdir(loveda_img_dir):
    if img_file.endswith(".png"):
        generate_sam_features(os.path.join(loveda_img_dir, img_file), sam_feature_dir)