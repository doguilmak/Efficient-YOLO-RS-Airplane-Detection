
<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

## YOLOv8 Models

The YOLOv8 training involved extensive experiments to assess different variants and optimization algorithms under controlled conditions. Each model was trained for **100 epochs** with a **fixed learning rate of 0.001** and a **batch size of 16**, using the **YOLOv8x**, **YOLOv8l**, and **YOLOv8s** variants. A total of **36** experiments tested various hyperparameter combinations, employing the optimizers **SGD**, **Adam**, and **AdamW**. Training data consisted of high-resolution satellite images, resized to **640x640** and **960x960** pixels, with adjustments made for color settings such as **hue**, **saturation**, **value**, and **mosaic**. Results indicated that models with a larger network size (960x960) consistently outperformed smaller variants across all metrics, particularly in mAP50-95, achieving values above 0.898. AdamW emerged as the most effective optimizer, yielding lower false positives and superior performance in YOLOv8l and YOLOv8x. **For detailed results and a comprehensive analysis of all experiments, please refer to the [Experiments Spreadsheet](https://github.com/doguilmak/Comprehensive-YOLO-Airplane-Detection/blob/main/2-Training/Experiments.xlsx).**

<br>

**Table 1. Table of Top 6 YOLOv8 Models Result.**

| Experiment ID | Model    | Hyperparameters                                                                                   | F1 Score | Precision | Recall | mAP50 | mAP50-95 | Weights |
|----------|----------|---------------------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|------------------|
| 12 | YOLOv8x  | Network size: 960x960<br>with Augmentation<br>Optimizer: SGD     | 0.9932   | 0.9915 | 0.9950 | 0.9939 | 0.8990 | [Download](https://link-to-weights.com/) |
| 32 | YOLOv8l  | Network size: 960x960<br>with Augmentation<br>Optimizer: AdamW   | 0.9930   | 0.9927 | 0.9933 | 0.9936 | 0.9025 | [Download](https://link-to-weights.com/) |
| 30 | YOLOv8l  | Network size: 960x960<br>with Augmentation<br>Optimizer: SGD     | 0.9922   | 0.9903 | 0.9940 | 0.9941 | 0.9021 | [Download](https://link-to-weights.com/) |
| 28 | YOLOv8l  | Network size: 960x960<br>with Augmentation<br>Optimizer: Adam    | 0.9921   | 0.9915 | 0.9928 | 0.9940 | 0.9018 | [Download](https://link-to-weights.com/) |
| 14 | YOLOv8x  | Network size: 960x960<br>with Augmentation<br>Optimizer: AdamW   | 0.9920   | 0.9915 | 0.9924 | 0.9938 | 0.9020 | [Download](https://link-to-weights.com/) |
| 50 | YOLOv8s  | Network size: 960x960<br>with Augmentation<br>Optimizer: AdamW   | 0.9918   | 0.9934 | 0.9903 | 0.9940 | 0.8983 | [Download](https://link-to-weights.com/) |

**Note:** Augmentation parameters include Hue (0.015), Saturation (0.7), Value (0.4), and Mosaic (1). For experiments without augmentation, all parameters are set to 0.

<br>

## YOLOv9e Models

To evaluate the performance differences between YOLOv8 and the "e" variant of the YOLOv9 architecture, an extensive set of experiments was carried out. These experiments tested the YOLOv9e architecture under various scenarios, utilizing different optimizers and data augmentation techniques. The training was conducted with a network size of 640x640, ensuring a fair comparison with the YOLOv8 models. A total of six distinct experiments were executed to determine the optimal model configurations. Each YOLOv9e model was trained for 100 epochs at a fixed learning rate of 0.001, and a batch size of 16 was consistently used across all experiments, in line with the settings from the YOLOv8 trials.

<br>

**Table 2. Comparison of YOLOv9e Models Result.**

| Experiment ID | Hyperparameters                                                                       | F1 Score | Precision | Recall | mAP50 | mAP50-95 | Weights |
|----------|---------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|---------------|
| 57 | Network size: 640x640<br>without Augmentation<br>Optimizer: SGD   | 0.9899 | 0.9912 | 0.9886 | 0.9935| 0.8982 | [Download](https://link-to-weights.com/) |
| 58 | Network size: 640x640<br>with Augmentation<br>Optimizer: SGD      | 0.9917 | 0.9918 | 0.9916 | 0.9937| 0.8989 | [Download](https://link-to-weights.com/) |
| 59 | Network size: 640x640<br>without Augmentation<br>Optimizer: Adam  | 0.9882 | 0.9864 | 0.9900 | 0.9930| 0.8954 | [Download](https://link-to-weights.com/) |
| 60 | Network size: 640x640<br>with Augmentation<br>Optimizer: Adam     | 0.9889 | 0.9885 | 0.9894 | 0.9934| 0.8886 | [Download](https://link-to-weights.com/) |
| 61 | Network size: 640x640<br>without Augmentation<br>Optimizer: AdamW | 0.9880 | 0.9864 | 0.9896 | 0.9930| 0.8954 | [Download](https://link-to-weights.com/) |
| 62 | Network size: 640x640<br>with Augmentation<br>Optimizer: AdamW    | 0.9899 | 0.9891 | 0.9907 | 0.9936| 0.8930 | [Download](https://link-to-weights.com/) |

<br>

##  Evaluation on CORS-ADD Dataset

We utilized the validation dataset from the CORS-ADD dataset to evaluate our HRPlanes-trained models without any prior training on CORS-ADD data. This approach allowed us to assess how well our models generalize to new datasets. Using the validation split of the CORS-ADD-HBB subset, we evaluated the performance of the top three YOLOv8 models and the top three YOLOv9e models. The HBB format was employed for annotating aircraft, providing a structured framework for detection.

The models demonstrated high precision in detecting aircraft under varying conditions. Notably, the evaluation highlighted that our models effectively retained the knowledge gained from training on other datasets while adapting to the unique features of the CORS-ADD dataset. This process ensured robust performance, validating the models' capabilities in diverse scenarios.

<br>

**Table 3. Performance Results of Top 3 YOLOv8 Models on the CORS-ADD Validation Data**

| Experiment ID | Model   | Hyperparameters                                                                       | F1 Score | Precision | Recall | mAP50 | mAP50-95 |
|----------|---------|---------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|
| 12 | YOLOv8x | Network size: 960x960<br>with Augmentation<br>Optimizer: SGD   | 0.8167 | 0.9033 | 0.7453 | 0.8441| 0.4499 |
| 32 | YOLOv8l | Network size: 960x960<br>with Augmentation<br>Optimizer: AdamW | 0.8060 | 0.8981 | 0.7311 | 0.8265| 0.4278 |
| 30 | YOLOv8l | Network size: 960x960<br>with Augmentation<br>Optimizer: SGD   | 0.8063 | 0.9014 | 0.7294 | 0.8272| 0.4239 |

The evaluation indicates that YOLOv8x with the SGD optimizer is the most effective configuration, achieving an F1 Score of 0.8167, Precision of 0.9033, and Recall of 0.7453. It also led in mAP50 with a score of 0.8441 and mAP50-95 at 0.4499, confirming its robustness and consistency compared to the YOLOv8l models.

<br>

**Table 4. Performance Results of Top 3 YOLOv9e Models on the CORS-ADD Validation Data**

| Experiment ID | Model   | Hyperparameters                                                                       | F1 Score | Precision | Recall | mAP50 | mAP50-95 |
|----------|---------|---------------------------------------------------------------------------------------|----------|-----------|--------|-------|----------|
| 58 | YOLOv9e | Network size: 640x640<br>with Augmentation<br>Optimizer: SGD    | 0.8169 | 0.9073 | 0.7428 | 0.8158| 0.4188 |
| 57 | YOLOv9e | Network size: 640x640<br>without Augmentation<br>Optimizer: SGD | 0.8107 | 0.9030 | 0.7355 | 0.8413| 0.4486 |
| 62 | YOLOv9e | Network size: 640x640<br>with Augmentation<br>Optimizer: AdamW  | 0.8096 | 0.8909 | 0.7419 | 0.8127| 0.3974 |


The model with the SGD optimizer and augmentation achieved the highest performance, with an F1 Score of 0.8169, Precision of 0.9073, and Recall of 0.7428. The subsequent model, using SGD without augmentation, had slightly lower metrics. Overall, the YOLOv9e model with SGD and augmentation demonstrated superior performance.

### Visual Comparison of Predictions

To complement the evaluation results, the accompanying figure presents a visual comparison between ground truth annotations and the predictions made by the best-performing YOLOv8 and YOLOv9 models using the CORS-ADD dataset. Images selected from the [CORS-ADD](https://ieeexplore.ieee.org/abstract/document/10144379) article. The figure is structured into three columns—representing Ground Truth (GT), YOLOv8 Predictions, and YOLOv9 Predictions—arranged in three rows. This layout effectively illustrates the differences and accuracies between the ground truth and the model predictions.

1. **Ground Truth Images**: Showcasing the HBB annotations for aircraft.
2. **YOLOv8x Predictions**: Utilizing a network size of 960x960 with an SGD optimizer and augmentation.
3. **YOLOv9e Predictions**: Using a network size of 640x640 with an SGD optimizer and similar augmentation.

This figure illustrates the performance of both models across various aircraft types and challenging conditions. YOLOv8x predictions closely align with ground truth, exhibiting high precision with fewer false positives and negatives. The YOLOv9e predictions are also effective but show subtle differences in bounding box placement, particularly in edge cases. This highlights the generalization capabilities of both models while revealing slight performance differences.

<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/gt_v8_v9_cropped.png" alt="HRPlanes and CORS-ADD Dataset Samples"/>
