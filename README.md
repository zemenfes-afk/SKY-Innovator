# 🌳 Forest Segmentation Project (Satellite → Drone Images)

This project builds a deep learning segmentation model using PyTorch to detect forest and deforestation areas.
The model is first trained using satellite images and masks, then adapted to drone images.



# Part 1: Training with Satellite Images

This project demonstrates how to build a **semantic segmentation model** that detects **Amazon rainforest regions** from satellite imagery.

The pipeline starts with **raw satellite `.tif` images and mask labels**, converts them to a clean dataset, and trains a segmentation model using **PyTorch** and **Segmentation Models PyTorch** inside **Google Colab**.



# 🛰️ Project Pipeline

The workflow of the project:

```
Satellite Images (.tif)
        ↓
Dataset Cleaning & Conversion
        ↓
Image + Mask Dataset
        ↓
DataLoader
        ↓
Segmentation Model Training
        ↓
Forest Mask Prediction
```

Architecture used:

* **U-Net**
* Encoder: **EfficientNet-B2**
* Framework: **PyTorch**



# 🚀 Run in Google Colab

This project is designed to run **directly in Google Colab**.

1️⃣ Open the notebook

```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook.ipynb
```

2️⃣ Enable GPU

```
Runtime → Change runtime type → GPU
```

3️⃣ Run the notebook cells sequentially.



# 📦 STEP 1 — Download Raw Data & Install Converter Tools

First install required libraries and download the dataset.

```python
!pip install -q kagglehub tifffile imagecodecs
```

Libraries used:

* **kagglehub** → download datasets from Kaggle
* **tifffile** → read `.tif` satellite images
* **Pillow** → image processing

This step prepares folders:

```
Clean_Amazon_Dataset/
   images/
   masks/
```



# 🔄 STEP 2 — Convert TIF Satellite Images to JPG

Satellite datasets are stored as `.tif` images.

This script converts them into **standard JPG images and PNG masks**.

Key processing steps:

✔ Fix image dimension order
✔ Remove extra channels
✔ Normalize brightness
✔ Convert masks to grayscale

Example conversion:

```
input  : satellite_image.tif
output : satellite_image.jpg
mask   : satellite_image.png
```

After conversion:

```
Clean_Amazon_Dataset/
   images/
       image_001.jpg
       image_002.jpg
   masks/
       image_001.png
       image_002.png
```



# 📊 Dataset Verification

Check whether images and masks match.

```python
img_count = len(os.listdir("Clean_Amazon_Dataset/images"))
mask_count = len(os.listdir("Clean_Amazon_Dataset/masks"))
```

Expected result:

```
Images found: N
Masks found:  N
SUCCESS: matching pairs
```

This ensures every image has a corresponding mask.



# 📂 STEP 3 — Dataset Loader

A custom dataset loader is created using **PyTorch**.

The dataset class:

```
SimpleAmazonDataset
```

Responsibilities:

* Load satellite image
* Load mask
* Resize to 256×256
* Convert to tensor
* Convert masks to binary

DataLoader configuration:

```
batch_size = 16
shuffle = True
```

Example batch visualization:

```
Image  → RGB satellite photo
Mask   → Forest segmentation mask
```

This step verifies that **data and labels align correctly**.



# 🧠 STEP 4 — Install Segmentation Model Library

Install the segmentation framework:

```python
!pip install segmentation-models-pytorch
```

This library provides ready-to-use architectures like:

* **U-Net**
* **DeepLabV3**
* **FPN**



# ⚙️ STEP 5 — Model Training

The segmentation model is trained with:

Model configuration:

```
Architecture: U-Net
Encoder: EfficientNet-B2
Input Channels: 3
Classes: 2
```

Classes represent:

```
0 → Background
1 → Forest
```

Loss functions:

* Dice Loss
* Cross Entropy Loss

Optimizer:

```
Adam
Learning Rate = 0.0001
```

Training runs for:

```
30 epochs
```

Evaluation metric:

**Intersection over Union**

Example training log:

```
Epoch 01 | Loss: 0.97 | IoU: 0.59
Epoch 10 | Loss: 0.38 | IoU: 0.91
Epoch 26 | Loss: 0.35 | IoU: 0.93
```

Best performing model is saved automatically.



# 🧪 STEP 6 — Visualizing Model Output

After training, the model predicts **forest masks** for satellite images.

Output example:

```
Input Satellite Image
        ↓
Predicted Forest Mask
        ↓
Highlighted Forest Regions
```

This helps visualize **deforestation detection or forest coverage**.



# 📈 Training Results

Example performance:

```
Best IoU: 0.9281
```

Meaning the predicted forest masks overlap **~93% with ground truth masks**.



# 📂 Project Structure

```
amazon-forest-segmentation
│
├── notebook.ipynb
├── Clean_Amazon_Dataset
│   ├── images
│   └── masks
│
├── best_amazon_model.pth
├── final_amazon_model.pth
│
└── README.md
```



# 🚁 Part 2: Drone Image Processing and Mask Generation

The final goal of this project is to **detect forest areas from drone imagery**.
Since drone images initially do not contain masks, we use the **trained satellite segmentation model to automatically generate masks**.



# STEP 1 — Check Drone Image Size

Before processing drone images, their dimensions are verified.

This step helps ensure that the drone images are compatible with the trained model.

Typical checks include:

* Image resolution
* Channel format (RGB)
* Image resizing if required

Example:

```
Drone Image Shape: (height, width, 3)
```



# STEP 2 — Generate Initial Drone Masks (M1–M4)

To test the approach, a small set of **sample drone images** is used.

The trained satellite model generates **predicted masks** for these images.

Example:

```
Drone Image → Model Prediction → Generated Mask
```

Sample outputs:

```
M1.png
M2.png
M3.png
M4.png
```

These masks help verify whether the model can **generalize from satellite imagery to drone imagery**.



# STEP 3 — Generate Masks for the Full Drone Dataset

After validating the masking process, the model is applied to the **entire drone dataset**.

In this project:

```
Total Drone Images: 392
```

The trained model automatically generates segmentation masks for each drone image.

Output dataset:

```
Drone_Dataset/
   images/
   masks/
```



# STEP 4 — Auto-Generate Masks Using the Satellite Model

The satellite-trained model is used as an **automatic mask generator**.

Pipeline:

```
Drone Image
     ↓
Satellite-trained U-Net Model
     ↓
Predicted Forest Mask
```

This step converts the drone dataset into a **fully labeled segmentation dataset**.



# STEP 5 — Model Testing on Drone Images

Finally, the model predictions are visualized on drone images.

Output visualization:

```
Drone Image
     ↓
Predicted Mask
     ↓
Forest Regions Highlighted
```

This demonstrates the model’s ability to **detect vegetation and forest areas from aerial drone imagery**.


These make the README look **much stronger for portfolios or research**.
