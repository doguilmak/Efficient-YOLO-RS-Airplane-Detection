
<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

## Introduction

The integration of artificial intelligence (AI) with Geographic Information Systems (GIS) has revolutionized geospatial data analysis, enabling more precise and efficient insights. This study focuses on urban area monitoring around Istanbul Airport (IST) using advanced deep learning techniques. By leveraging InferenceVision, we demonstrate a powerful framework capable of extracting and geolocating objects from high-resolution satellite imagery, addressing critical challenges in urban planning and environmental monitoring.

<br>

## The Role of InferenceVision

InferenceVision is at the core of this project, offering an innovative solution for geospatial intelligence. It processes high-resolution satellite images to detect objects, calculate their geographic coordinates, and produce results in an actionable format. This capability bridges the gap between traditional GIS methods and modern AI-driven approaches, enabling detailed analyses of dynamic environments like urban airports.

In this study, InferenceVision is tasked with the following:

1.  Image Processing and Object Detection:  
    High-resolution imagery of the Istanbul Airport area is analyzed to detect features of interest, such as aircraft, using the YOLOv8 deep learning model. Detected objects are annotated with bounding polygons.
2.  Geolocation:  
    Normalized coordinates of detected objects are converted into precise geographic coordinates (latitude and longitude) relative to the image's spatial bounds.
3.  Result Generation:  
    The output is saved as CSV files, enabling seamless integration into downstream analyses and decision-making workflows.

<br>

## Methodology

The process begins with the preparation of the required components, including YOLOv8 weights and satellite images. Using trained models with HRPlanes dataset and the InferenceVision, the satellite image is processed to generate predictions, which are filtered based on confidence scores to ensure accuracy. The outputs are systematically stored and exported for further evaluation.

The workflow emphasizes reproducibility and efficiency:

-   The execution time of the entire pipeline is measured to evaluate performance.
-   Filtered results are saved in easily accessible formats, ensuring data reliability for further insights.

<br>

## Findings and Contributions

The study illustrates the effectiveness of combining advanced deep learning models with geospatial data processing frameworks. Specifically, InferenceVision enhances geospatial intelligence by automating:

-   Feature detection and annotation in complex urban environments.
-   Accurate geolocation of detected features for enhanced GIS applications.

<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/IST_Exp12.png" alt="Detection results at IST"/>

In just **19.71 seconds**, with a network size of **800x960**, out of 93 planes in the scene, **86 planes** were successfully detected at Istanbul Airport (IST) along with their precise geographic coordinates.

