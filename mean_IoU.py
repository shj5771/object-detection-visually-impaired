import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO

# ⚙️ 설정
IMAGE_DIR = r'C:\Users\User\Desktop\street\street_10000\dataset\images\train'
LABEL_DIR = r'C:\Users\User\Desktop\street\street_10000\dataset\labels\train'
MODEL_PATH = r'C:\Users\User\PycharmProjects\PythonProject1\runs\segment\yolo11n-seg_finfin\weights\best.pt'
IOU_THRESHOLD = 0.5

# 🧠 IoU 계산 함수
def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

# 📌 polygon -> binary mask
def polygon_to_mask(polygon, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

# 📄 label txt -> (class, binary mask) 리스트
def load_label_file(label_path, img_shape):
    masks = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            coords[::2] = [int(x * img_shape[1]) for x in coords[::2]]  # x
            coords[1::2] = [int(y * img_shape[0]) for y in coords[1::2]]  # y
            mask = polygon_to_mask(coords, img_shape)
            masks.append((cls, mask))
    return masks

# 🎯 예측 vs 정답 비교 (한 이미지)
def compute_image_miou(pred_masks, pred_classes, gt_masks, gt_classes):
    matched_ious = []
    used_gt = set()
    for i, (pmask, pcls) in enumerate(zip(pred_masks, pred_classes)):
        best_iou = 0
        best_j = -1
        for j, (gmask, gcls) in enumerate(zip(gt_masks, gt_classes)):
            if j in used_gt or pcls != gcls:
                continue
            iou = compute_mask_iou(pmask, gmask)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= IOU_THRESHOLD:
            matched_ious.append(best_iou)
            used_gt.add(best_j)
    return np.mean(matched_ious) if matched_ious else 0.0

# 🚀 메인 함수
def main():
    model = YOLO(MODEL_PATH)
    label_files = sorted(glob(os.path.join(LABEL_DIR, '*.txt')))
    all_ious = []

    for label_path in label_files:
        filename = os.path.splitext(os.path.basename(label_path))[0]
        img_path = os.path.join(IMAGE_DIR, filename + '.jpg')

        if not os.path.exists(img_path):
            print(f"[⚠️] 이미지 없음: {img_path}")
            continue

        # 이미지 로드 및 크기
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 라벨 → 마스크
        gt_data = load_label_file(label_path, (h, w))
        if len(gt_data) == 0:
            continue
        gt_classes, gt_masks = zip(*gt_data)

        # YOLOv8 예측
        result = model(img)[0]
        if result.masks is None:
            continue

        # 마스크 리사이즈
        raw_pred_masks = result.masks.data.cpu().numpy().astype(bool)
        pred_masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                      for m in raw_pred_masks]

        pred_classes = result.boxes.cls.cpu().numpy().astype(int)

        # mIoU 계산
        miou = compute_image_miou(pred_masks, pred_classes, gt_masks, gt_classes)
        all_ious.append(miou)

        print(f"[✓] {filename} mIoU: {miou:.4f}")

    if all_ious:
        print(f"\n📊 전체 Mean IoU: {np.mean(all_ious):.4f}")
    else:
        print("\n⚠️ mIoU를 계산할 유효한 이미지가 없습니다.")

if __name__ == '__main__':
    main()
