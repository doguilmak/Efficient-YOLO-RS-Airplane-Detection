
<img src="https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

Following the previous experiments, a transfer learning approach was adopted by retraining the top three pretrained models on the CORS-ADD dataset. Each model was trained for 20 epochs using the CORS-ADD training set and evaluated on the validation data. This approach allowed the models to retain prior knowledge from other datasets while adapting to the specific features of CORS-ADD. By focusing exclusively on the CORS-ADD dataset during training and validation, the models improved their generalization to this domain, reduced overfitting, and achieved better performance in aircraft detection.

<br>

**Table 5. Performance Results of Top 3 YOLOv8 Models on the CORS-ADD Dataset Using Transfer Learning**

| Experiment ID | Model   | Hyperparameters                                                                       | F1 Score | Precision | Recall | mAP50 | mAP50-95 | Weights |
|----------|---------|---------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|---------------|
| 12 | YOLOv8x | Network size: 640x640<br>with Augmentation<br>Optimizer: SGD   | 0.9333 | 0.9579 | 0.9100 | 0.9503| 0.5931 | [Download](https://link-to-weights.com/) |
| 32 | YOLOv8l | Network size: 640x640<br>with Augmentation<br>Optimizer: AdamW | 0.9250 | 0.9499 | 0.9013 | 0.9425| 0.5678 | [Download](https://link-to-weights.com/) |
| 30 | YOLOv8l | Network size: 640x640<br>with Augmentation<br>Optimizer: SGD   | 0.9352 | 0.9586 | 0.9130 | 0.9505| 0.5824 | [Download](https://link-to-weights.com/) |

**Note:** Augmentation parameters include Hue (0.015), Saturation (0.7), Value (0.4), and Mosaic (1). For experiments without augmentation, all parameters are set to 0.

The comparison between the previous and current performance results highlights substantial improvements in model performance after applying transfer learning. For the YOLOv8x model, which initially had an F1 score of 0.8167, precision of 0.9033, and recall of 0.7453, transfer learning led to significant enhancements across all metrics. The F1 score increased by 11.3%, reaching 0.9333, while precision rose by 6.0% to 0.9579, and recall improved by 22.1%, reaching 0.9100. Similarly, mAP50 increased by 12.6%, from 0.8441 to 0.9503, and mAP50-95 saw a 31.8% gain, rising from 0.4499 to 0.5931. These improvements clearly demonstrate that transfer learning enhanced the model's ability to generalize and boosted its overall detection performance.

<br>

**Table 6. Performance Results of Top 3 YOLOv9e Models on the CORS-ADD Dataset Using Transfer Learning**

| Experiment ID | Model   | Hyperparameters                                                                       | F1 Score | Precision | Recall | mAP50 | mAP50-95 | Weights |
|----------|---------|---------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|---------------|
| 58 | YOLOv9e | Network size: 640x640<br>with Augmentation<br>Optimizer: SGD    | 0.9392 | 0.9560 | 0.9230 | 0.9526| 0.5942 | [Download](https://link-to-weights.com/) |
| 57 | YOLOv9e | Network size: 640x640<br>without Augmentation<br>Optimizer: SGD | 0.9304 | 0.9494 | 0.9121 | 0.9471| 0.5773 | [Download](https://link-to-weights.com/) |
| 62 | YOLOv9e | Network size: 640x640<br>with Augmentation<br>Optimizer: AdamW  | 0.9088 | 0.9452 | 0.8751 | 0.9255| 0.5239 | [Download](https://link-to-weights.com/) |

The comparison between the two sets of performance results for the YOLOv9e models, before and after applying transfer learning, reveals substantial improvements across most metrics. For the YOLOv9e model with SGD optimizer and data augmentation, the F1 score increased from 0.8169 to 0.9392, a 15.0% improvement. Precision rose from 0.9073 to 0.9560 (5.4%), and recall saw a substantial increase of 24.3%, going from 0.7428 to 0.9230. The mAP50 improved by 16.8%, from 0.8158 to 0.9526, while mAP50-95 saw a significant 41.9% increase, from 0.4188 to 0.5942.

<br>
