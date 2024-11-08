
<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

This repository includes weights and evaluation metrics for a range of YOLO architectures trained on high-resolution satellite imagery for airplane detection, utilizing both the HRPlanes and CORS-ADD datasets. The analysis explores both direct training and transfer learning techniques across various YOLO architectures. Models include YOLOv8 and YOLOv9 with training done via Ultralytics, respectively. Below are detailed metrics and download links for each model.

## Updates

**Comparative Assessment of YOLO Architectures for Airplane Detection Using VHR Satellite Imagery article is now available!**  
Explore and utilize these datasets to enhance your deep learning projects for airplane detection.

<details>
<summary>Latest updates...</summary>

<br>

**October 2024**  
- Comprehensive inference made on Chicago O'Hare International Airport (ORD/KORD), Amsterdam Schiphol Airport (AMS/EHAM), Beijing Capital International Airport (PEK/ZBAA), and Haneda International Airport (HND/RJTT) airports.

**September 2024**  
- Transfer learning models utilizing CORS-ADD data now included, improving generalization.

**June 2024**  
- Training process complete using YOLOv8 and YOLOv9 architectures.
  
**April 2024**  
- Pre-process stage complete. The hyperparameters were decided to make experiments.

</details>

<br>

## Datasets

### HRPlanes

<!-- <img src="https://raw.githubusercontent.com/RSandAI/HRPlanes/main/Assets/HRPlanes%20Samples%20All.png" alt="HRPlanes and CORS-ADD dataset samples"/> -->

The imagery required for the dataset was obtained from Google Earth. We downloaded 4800 x 2703 sized 3092 RGB images from major airports around the world, such as Paris-Charles de Gaulle, John F. Kennedy, Frankfurt, Istanbul, Madrid, Dallas, Las Vegas, and Amsterdam, as well as aircraft boneyards like Davis-Monthan Air Force Base. The dataset images were manually annotated by creating bounding boxes for each airplane using the HyperLabel software, which still provides annotation services as [Plainsight](https://app.plainsight.ai/). Quality control of each label was conducted through visual inspection by independent analysts who were not involved in the labeling procedure. A total of 18,477 airplanes have been labeled. A sample image and corresponding minimum bounding boxes for airplanes can be seen in the figure below. The dataset has been approximately split as 70% (2170 images) for training, 20% (620 images) for validation, and 10% (311 images) for testing. In addition, you can access the repository of the dataset in [here](https://github.com/TolgaBkm/HRPlanes). For more details on the dataset, please refer to the original article: [A benchmark dataset for deep learning-based airplane detection: HRPlanes](https://dergipark.org.tr/tr/pub/ijeg/issue/77206/1107890#article_cite).

#### Download The Dataset

The complete HRPlanes dataset is available in YOLO format. Access the dataset on [Google Drive](https://drive.google.com/drive/folders/1NYji6HWh4HRLQMTagsn4tTv4LOdDrc9P?usp=sharing) to explore airplane annotations across various global airports.

<br>

### CORS-ADD Dataset

The CORS-ADD dataset includes a diverse collection of images obtained from Google Earth and multiple satellites, such as WorldView-2, WorldView-3, Pleiades, Jilin-1, and IKONOS. A total of 7,337 images were manually annotated with horizontal bounding boxes (HBB) and oriented bounding boxes (OBB), resulting in 32,285 aircraft annotations. The dataset features a variety of scenes beyond aprons and runways, including aircraft carriers, oceans, and land with flying aircraft. It encompasses multiple types of aircraft, including civil aircraft, bombers, fighters, and early warning aircraft, with scales ranging from 4×4 to 240×240 pixels. 

Using the validation split of the CORS-ADD-HBB subset, we evaluated our models' performance. The test results were derived from the three most successful models from separate YOLOv9 and YOLOv8 experiments, demonstrating high precision in detecting aircraft under varying conditions. For more details on the dataset, please refer to the original article: [Complex optical remote-sensing aircraft detection dataset and benchmark](https://ieeexplore.ieee.org/abstract/document/10144379).

<br>

## Experimental Setup

The experiments were conducted using an **[NVIDIA A100 40GB SXM](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/2-Training/GPU/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)** GPU, which is equipped with 40GB of HBM2 memory and a memory bandwidth of 1,555 GB/s. This GPU supports 19.5 TFLOPS for both FP64 Tensor Core and FP32 computations and operates with a maximum thermal design power (TDP) of 400W (NVIDIA, 2021). The training environment was set up on Google Colab, utilizing CUDA version 12.2 to leverage GPU acceleration for model training and evaluation tasks.

<br>

## Results

### YOLOv8 Models

The YOLOv8 models were extensively trained and evaluated on the **HRPlanes dataset** to understand their performance across various configurations. We employed three different variants of YOLOv8: **YOLOv8x**, **YOLOv8l**, and **YOLOv8s**, with training conducted under controlled conditions over **100 epochs**, a fixed learning rate of **0.001**, and a batch size of **16**. A total of **36 experiments** were executed, exploring a wide range of hyperparameter combinations including optimizers such as **SGD**, **Adam**, and **AdamW**. Additionally, the models were tested using different image resolutions (640x640 and 960x960) and augmentation techniques (e.g., adjustments to hue, saturation, value, and mosaic).

The results indicated that models trained with **960x960 resolution** consistently outperformed their smaller counterparts, achieving higher mAP50-95 scores, particularly surpassing a value of **0.898**. Among the optimizers, **AdamW** was found to be the most effective, particularly for the larger variants YOLOv8l and YOLOv8x, delivering the best performance in terms of **mAP**, **precision**, and **recall** while reducing false positives. The **top six models** from these experiments were selected based on a comprehensive analysis of **mAP** and **F1 scores**. These models were then made available for download, offering a benchmark for further research and application. For a complete overview of these models and their configurations, please refer to [Table 1](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/2-Training) for further details.

### YOLOv9e Models

To assess potential improvements in performance, the **YOLOv9e** architecture was evaluated in parallel with YOLOv8, testing several optimizers and augmentation strategies. The experiments were conducted using a **640x640** resolution, consistent with the YOLOv8 trials for a fair comparison. Each model was trained for **100 epochs** under identical conditions (learning rate = 0.001, batch size = 16), allowing for an in-depth comparison of the two architectures.

Overall, YOLOv9e models achieved **competitive performance**, with **SGD** optimization and augmentation yielding the highest results in terms of **F1 Score**, **precision**, and **recall**. Notably, the YOLOv9e model with **augmentation** performed slightly better than the corresponding model without, suggesting that incorporating augmentation can enhance the generalization capabilities of the network. A detailed performance comparison of the YOLOv9e models can be found in [Table 2](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/2-Training).

### Evaluation on CORS-ADD Dataset

In addition to the primary experiments on the HRPlanes dataset, we conducted a evaluation using the **CORS-ADD dataset**. This evaluation tested how well the HRPlanes-trained models could generalize to a completely different dataset, without prior exposure to CORS-ADD data. The models were evaluated on the **CORS-ADD-HBB subset** using the **HBB annotation format**, and we focused on the top-performing YOLOv8 and YOLOv9e models.

The results demonstrated that the models retained their performance capabilities when applied to new data, exhibiting **high precision** and **robust detection** capabilities across different conditions. Notably, the **YOLOv8x** model with the **SGD optimizer** emerged as the most effective configuration in this cross-dataset evaluation, outperforming other models in terms of **F1 score**, **precision**, and **mAP**.  For a detailed performance breakdown of the top YOLOv8 and YOLOv9e models on the CORS-ADD dataset, please refer to [Tables 3 and 4](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/2-Training).

<br>

### Access to the Details

For those interested in a deeper analysis, all experimental configurations, results, and detailed performance metrics have been documented and made available through a comprehensive **[spreadsheet of experiment results](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/2-Training/Experiments.xlsx)**. This document contains all the specifics of the experiments conducted, including model hyperparameters, optimizer settings, and corresponding performance metrics, offering full transparency into the experimental process. Here you can find all the details about the training process: **[2-Training](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/2-Training).**

<br>

## Transfer Learning Using CORS-ADD Dataset

In this section, we explore the use of **transfer learning** to enhance the generalization capability of our models, specifically for **aircraft detection** using the CORS-ADD dataset. Transfer learning allows us to take advantage of previously trained models on the HRPlanes dataset and fine-tune them for optimal performance on the CORS-ADD dataset, which contains different characteristics and challenges.

### Methodology

We selected the top three models from the previous training experiments and performed **transfer learning** by retraining them for **20 epochs** on the **CORS-ADD training set**. This approach ensured that the models retained their learned features from the initial dataset while adapting to the new dataset’s unique features. We evaluated the performance of each model using the **CORS-ADD validation data**, focusing on key metrics such as **F1 score**, **precision**, **recall**, **mAP50**, and **mAP50-95**.

### Results

The transfer learning experiments led to significant improvements in model performance across all metrics. For instance, the **YOLOv8x** model, initially trained on HRPlanes, saw an **11.3% improvement** in **F1 score** (from 0.8167 to 0.9333), along with substantial increases in **precision** (+6.0%), **recall** (+22.1%), and **mAP50** (+12.6%). Similarly, the **YOLOv9e** model, with the SGD optimizer and data augmentation, exhibited a **15.0% increase** in **F1 score**, as well as notable gains in **precision** (+5.4%) and **recall** (+24.3%). 

These results demonstrate that **transfer learning** effectively boosts model performance by leveraging prior knowledge while adapting to new, domain-specific datasets.

<br>

### Access to the Details

To examine the detailed experimental setup, model configurations, and complete results of the transfer learning process, please refer to the full documentation available in the **[3-Transfer Learning](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/3-Transfer%20Learning).**

<br>

## Comprehensive Inference for Large Input Images

This section presents a thorough evaluation of the performance of a deep learning-based airplane detection model using **Very High Resolution (VHR)** satellite imagery from four major international airports: **Chicago O'Hare International Airport (ORD/KORD)**, **Amsterdam Schiphol Airport (AMS/EHAM)**, **Beijing Capital International Airport (PEK/ZBAA)**, and **Haneda International Airport (HND/RJTT)**. These airports were selected based on their high air traffic volume, availability of high-resolution imagery, and diversity in geographical and operational conditions. This ensures a comprehensive analysis of the model's performance across varied environments and operational scenarios.

### Methodology
The study used **VHR satellite imagery** with a spatial resolution of **0.31m** sourced from Google Satellites. To assess the model’s ability to perform at different scales, each airport image was segmented into three levels:
- **Level 1:** One large image covering the entire airport.
- **Level 2:** Four sections that divide the original image.
- **Level 3:** Sixteen smaller sections for more granular analysis.

The **YOLOv8x model**, previously trained on the HRPlanes dataset, was utilized for the inference process. The model was tested with input sizes of **640x640** and **960x960** pixels to evaluate how varying image resolutions impacted detection accuracy. Key performance metrics such as **precision**, **recall**, **F1 score**, and **mean average precision (mAP)** were recorded at both **mAP50** and **mAP50-95** thresholds.

<br>

### Access to the Details

For a more detailed look at the experimental setup, performance results, and the impact of image resolution and granularity on airplane detection accuracy, please refer to the full documentation available in the **[4-Comprehensive Inference](https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/tree/main/4-Comprehensive%20Inference)**.

<br>

## Citation

If this dataset or model weights benefit your research, please cite our [paper]().

<br>

## Copyright

The dataset and images are available for academic use only, adhering to [Google Earth’s terms](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).
