from ultralytics import YOLO
import os
import cv2

# Load your trained model
model      = YOLO("v8sbest.pt")

# Paths
image_dir  = r"D:/Kitchen Utensils/Kitchen Utensils Merged V3 20NOV24/extracted_images_Occluded 2"
output_dir = "predictions_txt_Occluded_2_conf_yolo"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_file)
    # Load image to get dimensions
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Run inference
    results = model(img_path)

    # Write out YOLO-format with confidence:
    # class_id conf x_center y_center width height
    base = os.path.splitext(img_file)[0]
    out_txt = os.path.join(output_dir, f"{base}.txt")
    with open(out_txt, "w") as f:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                # box.xywh = [x_center, y_center, box_w, box_h] in pixels
                x_c, y_c, bw, bh = box.xywh[0].tolist()
                # normalize
                x_c /= w
                y_c /= h
                bw  /= w
                bh  /= h

                f.write(f"{cls_id} {conf:.6f} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
