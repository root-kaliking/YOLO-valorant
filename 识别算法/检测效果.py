import cv2
import numpy as np
from ultralytics import YOLO
import mss

model = YOLO(r"D:\AI\ultralytics-8.3.163\runs\detect\valorant2.0\weights\best.pt")

def draw_boxes_only(image, boxes_info, line_width=2):

    img_copy = image.copy()
    color_map = {'body': (0, 255, 0), 'head': (0, 165, 255)}

    for box in boxes_info:
        x1, y1, x2, y2 = map(int, box[:4])
        label = box[4]
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_width)
    return img_copy

def strict_one_head_per_body(result):

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], [], [],

    body_boxes = []
    head_boxes = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        if cls_id == 0:
            body_boxes.append(bbox + ['body', conf])
        elif cls_id == 1:
            head_boxes.append(bbox + ['head', conf])

    body_boxes.sort(key=lambda b: b[5], reverse=True)
    head_boxes.sort(key=lambda h: h[5], reverse=True)
    paired_results = []
    used_head_indices = set()

    for body in body_boxes:
        bx1, by1, bx2, by2, body_label, body_conf = body
        body_width = bx2 - bx1
        body_height = by2 - by1
        body_center_x = (bx1 + bx2) / 2
        body_center_y = (by1 + by2) / 2

        best_head_idx = -1
        best_head_overlap = 0
        for head_idx, head in enumerate(head_boxes):
            if head_idx in used_head_indices:
                continue
            hx1, hy1, hx2, hy2, head_label, head_conf = head
            inter_x1 = max(bx1, hx1)
            inter_y1 = max(by1, hy1)
            inter_x2 = min(bx2, hx2)
            inter_y2 = min(by2, hy2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                head_area = (hx2 - hx1) * (hy2 - hy1)
                overlap_ratio = inter_area / head_area if head_area > 0 else 0
                head_center_y_val = (hy1 + hy2) / 2
                vertical_position = (by1 - head_center_y_val) / body_height if body_height > 0 else 0
                score = overlap_ratio + max(0, vertical_position) * 0.5 + head_conf * 0.2
                if score > best_head_overlap:
                    best_head_overlap = score
                    best_head_idx = head_idx

        head_to_use = None
        if best_head_idx != -1 and best_head_overlap > 0.3:
            head_to_use = head_boxes[best_head_idx]
            used_head_indices.add(best_head_idx)
        else:
            head_width = body_width * 0.6
            head_height = head_width * 1.1
            head_center_x = body_center_x
            head_x1 = head_center_x - head_width / 2
            head_x2 = head_center_x + head_width / 2
            head_x1 = max(bx1, head_x1)
            head_x2 = min(bx2, head_x2)
            head_width = head_x2 - head_x1
            max_head_bottom = by1 + body_height * 0.3
            head_y2 = min(max_head_bottom, by1 + head_height)
            head_y1 = max(by1 - head_height * 0.2, head_y2 - head_height)
            head_y1 = max(0, head_y1)
            generated_head = [head_x1, head_y1, head_x2, head_y2, 'head', body_conf * 0.6]
            head_to_use = generated_head

        paired_results.append((body, head_to_use))

    unpaired_heads = []
    for head_idx, head in enumerate(head_boxes):
        if head_idx not in used_head_indices:
            unpaired_heads.append(head)
    return paired_results, unpaired_heads, body_boxes, head_boxes

sct = mss.mss()

monitor = sct.monitors[1]

print("开始实时屏幕检测...")
print("操作说明:")
print("  1. 此窗口可以自由拖拽、缩放，请将其拖到iPad屏幕上。")
print("  2. 按 'Q' 键退出程序。")

# 创建一个可调整大小的显示窗口
window_name = 'Real-time Screen Detection (Drag to iPad)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.resizeWindow(window_name, 1280, 720)

while True:

    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model.predict(
        source=frame,
        save=False,
        show=False,
        conf=0.18,
        iou=0.4,
        imgsz=640,
        verbose=False
    )

    # 绘制结果框
    result_frame = frame.copy()
    for r in results:
        paired_results, unpaired_heads, body_boxes, head_boxes = strict_one_head_per_body(r)
        all_boxes_to_draw = []
        for body, head in paired_results:
            all_boxes_to_draw.append(body)
            if head is not None:
                all_boxes_to_draw.append(head)
        for head in unpaired_heads:
            all_boxes_to_draw.append(head)
        result_frame = draw_boxes_only(result_frame, all_boxes_to_draw, line_width=3)

    cv2.imshow(window_name, result_frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cv2.destroyAllWindows()
print("程序已退出。")