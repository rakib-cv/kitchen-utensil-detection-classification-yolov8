# kitchen-utensil-detection-classification-yolov8

A computer vision project for detecting and classifying kitchen utensils as a foundational step toward preventing kitchen accidents for diverse users, including children, elderly individuals, and people with physical disabilities.

## Overview

This project investigates the feasibility of using deep learning-based object detection for kitchen safety monitoring. A YOLOv8-based system was developed to detect and classify 20 categories of kitchen utensils under both occluded and non-occluded conditions. Each detected utensil is mapped to a static safety category, such as safe or dangerous based on utensil type.

This work is based on my research paper:

**Utensil Classification using Computer Vision: A Feasibility Study on Preventing Kitchen Accidents for Diverse Users**

## Motivation

Kitchen accidents frequently involve sharp or hazardous utensils such as knives and peelers. This project explores whether real-time object detection can serve as a foundation for future intelligent kitchen safety systems that support safer and more accessible cooking environments.

## Dataset

A custom dataset of kitchen utensils was constructed and manually annotated.

- **Number of classes:** 20
- **Total images:** 2,055
- **Train:** 1,596
- **Validation:** 309
- **Test:** 150

The dataset includes utensils captured under:
- **Occluded conditions**
- **Non-occluded conditions**

Examples of classes include:
- knife
- spoon
- ladle
- peeler
- tongs
- spatula
- serving spoon
- pizza cutter
- masher

## Method

The detection pipeline is based on **YOLOv8**, chosen for its real-time performance and strong detection accuracy.

### Pipeline
1. Input video frame
2. YOLOv8 detects utensils
3. Bounding box, class label, and confidence score are assigned
4. Each utensil is mapped to a predefined risk category:
   - **Dangerous**: e.g. knives, peelers, tongs
   - **Safe**: e.g. ladles, spatulas, spoons
5. Output is visualized with annotated frames

## Training Configuration

- **Image size:** 640 × 640
- **Learning rate:** 0.01
- **Momentum:** 0.937
- **Batch size:** 16
- **Epochs:** 200

## Models Evaluation

The following YOLOv8 variants were compared:

- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLOv8l

### Performance on custom dataset

| Model   | F1 Score | Precision | Recall | mAP50 |
|---------|----------|-----------|--------|-------|
| YOLOv8n | 0.95     | 0.997     | 0.99   | 0.979 |
| YOLOv8s | 0.96     | 0.995     | 0.99   | 0.982 |
| YOLOv8m | 0.96     | 0.994     | 0.99   | 0.983 |
| YOLOv8l | 0.96     | 0.983     | 0.99   | 0.984 |

YOLOv8s was identified as a strong balance between speed and accuracy for real-time kitchen monitoring.

## Evaluation on Kitchen Scenes

The system was further tested on two scene settings:

| Dataset        | Images | Precision | Recall | F1 Score | mAP@50 | mAP@50-95 |
|----------------|--------|-----------|--------|----------|--------|-----------|
| Occluded       | 200    | 0.87      | 0.93   | 0.90     | 0.89   | 0.72      |
| Non-occluded   | 200    | 0.84      | 0.97   | 0.90     | 0.93   | 0.80      |

These results show that the system remains effective even under occlusion, though localization becomes more difficult in more challenging scenes.

## Key Contributions

- Built a custom kitchen utensil dataset with 20 classes
- Trained and evaluated YOLOv8 for utensil detection
- Compared multiple YOLOv8 model variants
- Evaluated performance under occluded and non-occluded conditions
- Introduced a safety-oriented static risk mapping for utensil categories

## Limitations

This study was conducted in a controlled environment and does not yet include:
- real kitchen deployment
- human-object interaction modeling
- dynamic risk estimation
- real-time alert generation

## Future Work

Future work will focus on:
- real-world kitchen validation
- dynamic risk modeling using motion and context
- human-hand interaction analysis
- assistive safety alerts for vulnerable users
- deployment on edge or embedded systems


The following examples illustrate detection performance under different visibility conditions.

<h2>Sample Results</h2>

<p>Example detections under occluded and non-occluded kitchen scenes:</p>

<table align="center">
  <tr>
    <th>Occluded Scene</th>
    <th>Non-Occluded Scene</th>
  </tr>
  <tr>
    <td align="center">
      <img src="images/occluded_example.jpg" width="400"/>
    </td>
    <td align="center">
      <img src="images/non_occluded_example.jpg" width="400"/>
    </td>
  </tr>
</table>

## Tech Stack

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Roboflow
- CUDA
- PyTorch

## Citation

Published at ICIIBMS 2025 (IEEE Conference)

If you use or reference this work, please cite:

```bibtex
@inproceedings{rakib2025utensil,
  title={Utensil Classification using Computer Vision: A Feasibility Study on Preventing Kitchen Accidents for Diverse Users},
  author={Rakib, Nma Salam and Karungaru, Stephen and Terada, Kenji and Hasan, Mohammad Mehedi},
  booktitle={2025 10th International Conference on Intelligent Informatics and Biomedical Sciences (ICIIBMS)},
  volume={10},
  pages={318--325},
  year={2025},
  organization={IEEE}
}

##  Author

- **Rakib NMA Salam
- **PhD Researcher, Tokushima University  
- **Computer Vision, Object Detection, Kitchen Safety AI
