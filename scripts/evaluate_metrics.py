import os
import re
import csv
import numpy as np
from collections import defaultdict

TXT_FOLDER = r"D:/Kitchen Utensils/Kitchen Utensils Merged V3 20NOV24/gt_pred_occluded_conf"
CSV_OUT = "final_detection_evaluation_occluded 2.csv"

def load_gt_boxes(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = line.strip().split()
            boxes.append({
                "class_id": int(cls),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h)
            })
    return boxes

def load_pred_boxes(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                cls, conf, xc, yc, w, h = parts
            elif len(parts) == 5:
                cls, xc, yc, w, h = parts
                conf = 1.0
            else:
                continue
            boxes.append({
                "class_id": int(cls),
                "conf": float(conf),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h)
            })
    return boxes

def yolo_to_bbox(box):
    xc, yc, w, h = box["xc"], box["yc"], box["w"], box["h"]
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

def compute_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idxs = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idxs + 1] - mrec[idxs]) * mpre[idxs + 1])

def evaluate_all(folder, iou_thresh=0.5):
    files = os.listdir(folder)
    pred_files = sorted(f for f in files if re.fullmatch(r"frame_\d+\.txt", f))

    per_frame = []
    gt_dict = {}
    pred_list = []
    total_gt_per_class = defaultdict(int)

    for pf in pred_files:
        base = os.path.splitext(pf)[0]
        gt_candidates = [g for g in files if g.startswith(base + "_png.rf") and g.endswith(".txt")]
        if not gt_candidates:
            print(f"[!] skipping {pf}: no GT file")
            continue
        gt_file = gt_candidates[0]

        pboxes = load_pred_boxes(os.path.join(folder, pf))
        gboxes = load_gt_boxes(os.path.join(folder, gt_file))
        gt_dict[base] = gboxes

        for g in gboxes:
            total_gt_per_class[g["class_id"]] += 1

        matched = set()
        tp = 0
        ious = []

        for pb in pboxes:
            best_iou, best_j = 0, None
            bbox_p = yolo_to_bbox(pb)
            for j, gb in enumerate(gboxes):
                if j in matched or gb["class_id"] != pb["class_id"]:
                    continue
                bbox_g = yolo_to_bbox(gb)
                cur = compute_iou(bbox_p, bbox_g)
                if cur > best_iou:
                    best_iou, best_j = cur, j
            if best_iou >= iou_thresh:
                tp += 1
                matched.add(best_j)
                ious.append(best_iou)

            pred_list.append({
                "frame": base,
                "class_id": pb["class_id"],
                "conf": pb["conf"],
                "bbox": bbox_p
            })

        fp = len(pboxes) - tp
        fn = len(gboxes) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        miou = np.mean(ious) if ious else 0

        per_frame.append({
            "frame": base,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": f"{prec:.2f}",
            "Recall": f"{rec:.2f}",
            "F1": f"{f1:.2f}",
            "Mean_IoU": f"{miou:.2f}"
        })

    total_tp = sum(int(f["TP"]) for f in per_frame)
    total_fp = sum(int(f["FP"]) for f in per_frame)
    total_fn = sum(int(f["FN"]) for f in per_frame)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    mean_iou = np.mean([float(f["Mean_IoU"]) for f in per_frame]) if per_frame else 0

    print(f"Evaluated {len(per_frame)} frames")
    print(f" TP = {total_tp}, FP = {total_fp}, FN = {total_fn}")
    print(f" Precision = {precision:.2f}")
    print(f" Recall    = {recall:.2f}")
    print(f" F1 Score  = {f1:.2f}")
    print(f" Mean IoU  = {mean_iou:.2f}")

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map_results = []

    pred_list.sort(key=lambda x: x["conf"], reverse=True)
    for iou_t in iou_thresholds:
        APs = []
        for cls_id, n_gt in total_gt_per_class.items():
            matched = set()
            tps = []
            for p in pred_list:
                if p["class_id"] != cls_id:
                    continue
                best_iou, best_j = 0, None
                for idx, g in enumerate(gt_dict[p["frame"]]):
                    if g["class_id"] != cls_id or (p["frame"], idx) in matched:
                        continue
                    cur = compute_iou(p["bbox"], yolo_to_bbox(g))
                    if cur > best_iou:
                        best_iou, best_j = cur, idx
                if best_iou >= iou_t:
                    tps.append(1)
                    matched.add((p["frame"], best_j))
                else:
                    tps.append(0)
            if not tps:
                continue
            fps = [1 - x for x in tps]
            tps_cum = np.cumsum(tps)
            fps_cum = np.cumsum(fps)
            precisions = tps_cum / (tps_cum + fps_cum + 1e-16)
            recalls = tps_cum / n_gt
            APs.append(compute_ap(recalls, precisions))
        map_results.append(np.mean(APs) if APs else 0)

    mAP_50 = map_results[0]
    mAP_5095 = np.mean(map_results)

    print(f"Global mAP@0.5 = {mAP_50:.2f}")
    print(f"Global mAP@0.5:0.95 = {mAP_5095:.2f}")

    for row in per_frame:
        row["mAP@0.5"] = ""
        row["mAP@0.5:0.95"] = ""

    per_frame.append({
        "frame": "OVERALL",
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": f"{precision:.2f}",
        "Recall": f"{recall:.2f}",
        "F1": f"{f1:.2f}",
        "Mean_IoU": f"{mean_iou:.2f}",
        "mAP@0.5": f"{mAP_50:.2f}",
        "mAP@0.5:0.95": f"{mAP_5095:.2f}"
    })

    with open(CSV_OUT, "w", newline="") as f:
        fields = ["frame", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean_IoU", "mAP@0.5", "mAP@0.5:0.95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)

    print(f">>> Full results written to {CSV_OUT}")

# Run
if __name__ == "__main__":
    evaluate_all(TXT_FOLDER)
