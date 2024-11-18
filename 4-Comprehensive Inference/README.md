
<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

This study presents a comprehensive evaluation of a deep learning-based plane detection model using high-resolution satellite imagery from four major international airports: [Chicago O'Hare International Airport (ORD/KORD)](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/4-Comprehensive%20Inference/ORD), [Amsterdam Schiphol Airport (AMS/EHAM)](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/4-Comprehensive%20Inference/AMS), [Beijing Capital International Airport (PEK/ZBAA)](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/4-Comprehensive%20Inference/PEK), and [Haneda International Airport (HND/RJTT)](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/4-Comprehensive%20Inference/HND). These airports were selected based on several criteria, including high air traffic volume, the availability of high-resolution imagery, and their representation of diverse geographical and operational conditions. This selection ensures a robust assessment of the model's performance across different environmental factors and operational scenarios.

To rigorously assess the model's efficacy, we conducted a series of 36 experiments, varying key factors such as image resolution, model architecture, and network size. Each experiment was designed to evaluate the model’s performance under different conditions, ultimately identifying the configuration that delivers the best results for plane detection in satellite imagery. **For detailed results and a comprehensive analysis of all experiments, please refer to the [Experiments Spreadsheet](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/4-Comprehensive%20Inference/Inference%20Results.xlsx).**

### Methodology
Each Very High Resolution (VHR) satellite image has a spatial resolution of 0.31m, sourced from Google Satellites. The original image has a resolution of 8000x8000 pixels. To analyze the model's performance at various scales, each image was segmented into three levels of granularity:
- **Level 1:** A single large image covering the airport area.
- **Level 2:** Four sections subdividing the original image.
- **Level 3:** Sixteen smaller sections for a finer-grain analysis.

The inference utilized the best-performing model, YOLOv8x, trained on the HRPlanes dataset. This model was evaluated with input sizes of 640x640, 960x960, and 1280x1280 pixels to determine how image resolution impacts detection performance. Key metrics, including precision, recall, F1 score, and mean average precision (mAP) at thresholds of mAP50 and mAP50-95, were recorded to quantify detection accuracy.

<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/ci_ORD.png" alt="Comprehensive Inference for Large Input Images"/>

Figure illustrates the results of airplane detection at Chicago O'Hare International Airport (ORD/KORD) using the YOLOv8x model with a 960x960 pixel network input size. The analysis is performed across three levels of image granularity: Level 1 (a), Level 2 (b), and Level 3 (c).

<br>

### Key Findings

The results indicate that model performance varied significantly across different airports, image levels, and input sizes:
- **Top Performance:** Experiment 32 (PEK/ZBAA, Level 2, 960x960) achieved near-perfect detection with an F1 Score of 0.9992, Precision of 0.9984, and Perfect Recall (1.0000), showcasing exceptional reliability in airplane identification. Experiment 34 (PEK/ZBAA, Level 1, 1280x1280) also delivered outstanding results with an F1 Score of 0.9991, Perfect Precision (1.0000), and high Recall of 0.9982, further emphasizing its strong performance across multiple metrics.
- **Lower Performance:** At ORD/KORD (Level 1, 640x640), the model’s F1 Score dropped to 0.7388 with a Recall of 0.6047, detecting only 68 airplanes out of 129, likely due to visibility limitations in lower-resolution images. Similar performance challenges were noted at PEK/ZBAA (Level 1, 640x640), where Precision was 0.7778.
- **Granularity Insights:** The results highlight a general trend of improved detection accuracy in finer image levels where individual airplanes are more distinguishable. Larger image levels often posed difficulties, likely due to partial airplane occlusion or double tagging, affecting the precision of airplane counts.

<br>

**Table 7. Top 6 Results of the Comprehensive Inference**

| Exp. No | IATA/ICAO Code | Image Level | Network Size | Number of Airplanes (GT) | Number of Airplanes (Inference) | F1 Score | Precision | Recall | mAP50 | mAP50-95 |
|---------|-----------------|-------------|--------------|--------------------------|---------------------------------|----------|-----------|--------|-------|----------|
| 32      | PEK/ZBAA        | 2           | 960x960      | 31                       | 31                              | 0.9992   | 0.9984    | 1      | 0.995 | 0.7854   |
| 34      | PEK/ZBAA        | 1           | 1280x1280    | 31                       | 30                              | 0.9991   | 1         | 0.9982 | 0.995 | 0.7741   |
| 25      | AMS/EHAM        | 1           | 1280x1280    | 74                       | 74                              | 0.9931   | 0.9862    | 1      | 0.9947 | 0.8303   |
| 6       | ORD/KORD        | 3           | 960x960      | 131                      | 126                             | 0.9876   | 1         | 0.9754 | 0.9911 | 0.8044   |
| 13      | HND/RJTT        | 1           | 960x960      | 61                       | 60                              | 0.9899   | 0.9963    | 0.9836 | 0.9944 | 0.7617   |
| 17      | HND/RJTT        | 2           | 1280x1280    | 64                       | 61                              | 0.9837   | 1         | 0.9678 | 0.9833 | 0.8113   |


*Note: Full results are provided for all experiments, capturing the impact of image granularity, resolution, and model input size on airplane detection accuracy.*

These findings underline the importance of image resolution and granularity in VHR imagery analysis, suggesting that finer subdivisions and higher resolutions are more effective for accurate object detection in large-scale images.
