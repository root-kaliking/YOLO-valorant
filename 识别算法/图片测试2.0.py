import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\AI\ultralytics-8.3.163\runs\detect\valorant2.0\weights\best.pt")

# 进行预测
results = model.predict(
    source=r"C:\Users\王泓正\OneDrive\图片\Screenshots\屏幕截图 2025-11-27 140728.png",
    save=False,
    show=False,
    conf=0.18,
    iou=0.4,
    imgsz=640,
    verbose=False
)


def draw_boxes_only(image, boxes_info, line_width=2):
    img_copy = image.copy()
    color_map = {'body': (0, 255, 0), 'head': (0, 165, 255)}  # 身体绿色，头橙色

    for box in boxes_info:
        x1, y1, x2, y2 = map(int, box[:4])
        label = box[4]
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_width)

    return img_copy


def strict_one_head_per_body(result):

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], []

    # 分离身体和头部检测结果
    body_boxes = []
    head_boxes = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        if cls_id == 0:  # 身体
            body_boxes.append(bbox + ['body', conf])
        elif cls_id == 1:  # 头部
            head_boxes.append(bbox + ['head', conf])

    # 按照置信度排序，优先处理高置信度的检测
    body_boxes.sort(key=lambda b: b[5], reverse=True)
    head_boxes.sort(key=lambda h: h[5], reverse=True)

    # 配对结果：每个身体对应一个头部（或None）
    paired_results = []
    used_head_indices = set()

    # 为每个身体框寻找最匹配的头部框
    for body_idx, body in enumerate(body_boxes):
        bx1, by1, bx2, by2, body_label, body_conf = body
        body_width = bx2 - bx1
        body_height = by2 - by1

        # 计算身体框的中心和区域
        body_center_x = (bx1 + bx2) / 2
        body_center_y = (by1 + by2) / 2

        # 寻找最适合当前身体的头部框
        best_head_idx = -1
        best_head_overlap = 0  # 头部与身体的重叠度

        for head_idx, head in enumerate(head_boxes):
            if head_idx in used_head_indices:
                continue

            hx1, hy1, hx2, hy2, head_label, head_conf = head

            # 计算头部框在身体框内部的比例
            # 头部框与身体框的交集
            inter_x1 = max(bx1, hx1)
            inter_y1 = max(by1, hy1)
            inter_x2 = min(bx2, hx2)
            inter_y2 = min(by2, hy2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                # 计算交集面积
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                head_area = (hx2 - hx1) * (hy2 - hy1)

                # 头部在身体内部的比例
                overlap_ratio = inter_area / head_area if head_area > 0 else 0

                # 额外的约束：头部应该在身体的上半部分
                head_center_y = (hy1 + hy2) / 2
                vertical_position = (by1 - head_center_y) / body_height if body_height > 0 else 0

                # 综合评分：重叠比例 + 垂直位置 + 置信度
                score = overlap_ratio + max(0, vertical_position) * 0.5 + head_conf * 0.2

                if score > best_head_overlap:
                    best_head_overlap = score
                    best_head_idx = head_idx

        # 决定是否使用检测到的头部框
        head_to_use = None

        if best_head_idx != -1 and best_head_overlap > 0.3:
            # 使用检测到的头部框
            head_to_use = head_boxes[best_head_idx]
            used_head_indices.add(best_head_idx)
        else:
            # 生成一个在身体内部的头部框
            # 头部宽度为身体的60%
            head_width = body_width * 0.6
            head_height = head_width * 1.1

            # 头部中心与身体中心水平对齐
            head_center_x = body_center_x

            # 计算头部框的边界
            head_x1 = head_center_x - head_width / 2
            head_x2 = head_center_x + head_width / 2

            # 确保头部框完全在身体框内部（水平方向）
            head_x1 = max(bx1, head_x1)
            head_x2 = min(bx2, head_x2)

            # 调整头部宽度
            head_width = head_x2 - head_x1

            # 垂直位置：头部在身体的上半部分
            # 头部底部不超过身体的30%高度位置
            max_head_bottom = by1 + body_height * 0.3

            # 头部顶部至少离身体顶部10像素
            head_y2 = min(max_head_bottom, by1 + head_height)
            head_y1 = max(by1 - head_height * 0.2, head_y2 - head_height)  # 确保头部高度

            # 确保头部框不会超出图像边界
            head_y1 = max(0, head_y1)

            # 创建生成的头部框
            generated_head = [head_x1, head_y1, head_x2, head_y2, 'head', body_conf * 0.6]
            head_to_use = generated_head

        # 保存配对结果
        paired_results.append((body, head_to_use))

    # 处理未配对的头部（独立的头部检测）
    unpaired_heads = []
    for head_idx, head in enumerate(head_boxes):
        if head_idx not in used_head_indices:
            unpaired_heads.append(head)

    return paired_results, unpaired_heads, body_boxes, head_boxes


# 主处理循环
for r_idx, r in enumerate(results):
    orig_img = r.orig_img if hasattr(r, 'orig_img') else r.plot()

    # 应用配对算法
    paired_results, unpaired_heads, body_boxes, head_boxes = strict_one_head_per_body(r)

    # 打印检测统计
    print(f"\n=== 第 {r_idx + 1} 张图片 ===")
    print(f"检测到 {len(body_boxes)} 个身体框, {len(head_boxes)} 个头部框")
    print(f"成功配对 {len(paired_results)} 个身体框")
    print(f"未配对的头部框: {len(unpaired_heads)} 个")

    # 准备所有要绘制的框
    all_boxes_to_draw = []

    # 添加配对的身体和头部框
    for body, head in paired_results:
        all_boxes_to_draw.append(body)
        if head is not None:
            all_boxes_to_draw.append(head)

    # 添加未配对的头部框（作为独立检测）
    for head in unpaired_heads:
        all_boxes_to_draw.append(head)

    # 绘制结果（只绘制框，不绘制标签）
    result_img = draw_boxes_only(orig_img, all_boxes_to_draw, line_width=3)

    # 调整窗口大小确保完整显示
    cv2.namedWindow(f"Detection Result {r_idx + 1}", cv2.WINDOW_NORMAL)

    # 获取图像尺寸
    img_height, img_width = result_img.shape[:2]

    # 计算合适的显示尺寸（不超过屏幕的80%）
    screen_width = 1920
    screen_height = 1080

    max_display_width = int(screen_width * 0.8)
    max_display_height = int(screen_height * 0.8)

    # 如果需要缩放
    if img_width > max_display_width or img_height > max_display_height:
        width_scale = max_display_width / img_width
        height_scale = max_display_height / img_height
        scale = min(width_scale, height_scale)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        result_img = cv2.resize(result_img, (new_width, new_height))

    # 显示结果
    cv2.imshow(f"Detection Result {r_idx + 1}", result_img)

    # 调整窗口大小以适应图像
    cv2.resizeWindow(f"Detection Result {r_idx + 1}", result_img.shape[1], result_img.shape[0])

    print("\n操作说明:")
    print("  - 按 's' 键保存图片")
    print("  - 按 'd' 键查看下一张（如果有）")
    print("  - 按 'q', 'ESC' 或关闭窗口退出")

    key = cv2.waitKey(0)

    if key == ord('s') or key == ord('S'):
        save_path = f"strict_one_head_per_body_{r_idx + 1}.jpg"
        cv2.imwrite(save_path, result_img)
        print(f"已保存到: {save_path}")
    elif key == ord('d') or key == ord('D'):
        print("继续下一张...")
        cv2.destroyAllWindows()
        continue
    elif key == 27 or key == ord('q') or key == ord('Q'):
        print("用户退出")
        break

    cv2.destroyAllWindows()

print(f"\n处理完成! 共处理 {len(results)} 张图片")