
<img src="https://github.com/RSandAI/Comprehensive-YOLO-Airplane-Detection/blob/main/assets/image.png" height=450 width=1280 alt=""/>

<br>

In this section, we detail the steps taken to prepare the dataset for use with the YOLO (You Only Look Once) object detection model. The goal of this pre-processing phase is to organize the data into appropriate subsets and ensure that both image and label files are correctly structured for training, validation, and testing.

### Organizing the Dataset

We began by creating the necessary folder structure for the dataset. Specifically, we set up three main directories: **train**, **validation**, and **test**. Within each of these directories, we created two subdirectories:

- **images**: to store the image files (e.g., `.jpg`)
- **labels**: to store the corresponding annotation files (e.g., `.txt`)

This folder structure is essential for the YOLO model, as it expects the data to be organized in this way.

### Splitting the Data

The next step involved organizing the data into the `train`, `validation`, and `test` sets based on the filenames listed in the `train.txt`, `validation.txt`, and `test.txt` files. These text files contain the list of image and label filenames, and we used them to copy the corresponding files into the appropriate folders.

- **Training set**: Used for training the model.
- **Validation set**: Used to evaluate the model during training.
- **Test set**: Used to evaluate the model’s performance after training is complete.

By following this process, we ensured that each subset (train, validation, and test) contains the correct images and their associated label files, making the dataset ready for model training.

### Verifying the Data

After organizing the files, we performed a check to ensure that all images were correctly placed into their respective directories. The number of image files in each subset was counted and visualized in a bar plot, providing a clear overview of the dataset distribution.

### Finalizing the Dataset Location

Once the data was correctly organized, we moved the pre-processed dataset to the final location on Google Drive. This makes the dataset easily accessible for further use, such as training the YOLO model.

### Summary

At the conclusion of this pre-processing phase, the dataset was fully organized into **train**, **validation**, and **test** sets, each containing images and corresponding labels in the required folder structure. This ensures that the data is in the proper format for training the YOLO model, and is now ready for the next steps in the machine learning pipeline.
