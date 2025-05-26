import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 원본 경로를 UTF-8로 인코딩
# image_dir = os.fsencode(r"C:\Users\User\Desktop\orgpic")

# 원본 경로
image_dir = r"C:\Users\User\Desktop\traffic\traffic_light\images"
label_dir = r"C:\Users\User\Desktop\traffic\traffic_light\labels"

# 저장 경로
aug_img_dir = r"C:\Users\User\Desktop\traffic\traffic_light\augmented_images"
aug_label_dir = r"C:\Users\User\Desktop\traffic\traffic_light\augmented_labels"
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# 증강 함수 정의
def save_augmented(image, label_path, save_name, transform_func, coord_func=None):
    aug_img = transform_func(image)
    aug_img_path = os.path.join(aug_img_dir, f"{save_name}.jpg")
    cv2.imwrite(aug_img_path, aug_img)

    # 라벨 처리
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            class_id = parts[0]
            coords = list(map(float, parts[1:]))

            if coord_func:
                coords = coord_func(coords)
            coord_str = " ".join(f"{x:.6f}" for x in coords)
            new_lines.append(f"{class_id} {coord_str}")

        aug_label_path = os.path.join(aug_label_dir, f"{save_name}.txt")
        with open(aug_label_path, 'w') as f:
            f.write("\n".join(new_lines))

# 좌우반전 좌표 변환
def flip_coords(coords):
    return [(1 - coords[i]) if i % 2 == 0 else coords[i] for i in range(len(coords))]

# Resize는 좌표 그대로 (정규화된 좌표 기준이면)
def identity_coords(coords):
    return coords

# 이미지 증강 함수들
def flip_image(img): return cv2.flip(img, 1)
def resize_image(img): return cv2.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)))
def brighten_image(img): return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
def contrast_image(img): return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
def blur_image(img): return cv2.GaussianBlur(img, (5, 5), 0)
def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# 전체 처리
for fname in os.listdir(image_dir):
    if fname.endswith(".jpg"):
        name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, f"{name}.txt")
        img = cv2.imread(image_path)

        save_augmented(img, label_path, f"{name}_flip", flip_image, flip_coords)
        save_augmented(img, label_path, f"{name}_resize", resize_image, identity_coords)
        save_augmented(img, label_path, f"{name}_bright", brighten_image, identity_coords)
        save_augmented(img, label_path, f"{name}_contrast", contrast_image, identity_coords)
        save_augmented(img, label_path, f"{name}_blur", blur_image, identity_coords)
        save_augmented(img, label_path, f"{name}_sharp", sharpen_image, identity_coords)
