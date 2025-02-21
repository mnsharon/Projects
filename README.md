# Project
# Skin Cancer Detection using Deep Learning

## Overview
This project uses **Convolutional Neural Networks (CNNs)** to classify skin lesions as benign or malignant, improving early diagnosis and treatment efficiency.

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Jupyter Notebook**

## Installation
To run this project, install the required dependencies using the following command:
```bash
pip install tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run
1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. **Set Up Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
5. **Open and Execute** `Skin_Cancer_Detection.ipynb`
   - This will download datasets, preprocess images, train the CNN model, and evaluate performance.

## Import Required Libraries
```python
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import get_file
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score
import os, glob, zipfile, random
```

## Set Random Seeds for Reproducibility
```python
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)
```

## Dataset Preparation
```python
train_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip"
valid_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip"
test_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip"

def download_and_extract_dataset():
    for i, download_link in enumerate([valid_url, train_url, test_url]):
        temp_file = f"temp{i}.zip"
        data_dir = get_file(origin=download_link, fname=os.path.join(os.getcwd(), temp_file))
        print("Extracting", download_link)
        with zipfile.ZipFile(data_dir, "r") as z:
            z.extractall("data")
        os.remove(temp_file)

download_and_extract_dataset()
```

## Generate CSV Metadata
```python
def generate_csv(folder, label2int):
    folder_name = os.path.basename(folder)
    labels = list(label2int)
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df.loc[i] = [filepath, label2int[label]]
            i += 1
    df.to_csv(f"{folder_name}.csv", index=False)

generate_csv("data/train", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_csv("data/valid", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_csv("data/test", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
```

## Data Preprocessing
```python
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [299, 299])

def process_path(filepath, label):
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, label
```

## Load and Process Data
```python
df_train = pd.read_csv("train.csv")
df_valid = pd.read_csv("valid.csv")
train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath"], df_train["label"])).map(process_path)
valid_ds = tf.data.Dataset.from_tensor_slices((df_valid["filepath"], df_valid["label"])).map(process_path)
```

## Model Training
```python
module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = m.fit(train_ds.batch(64), validation_data=valid_ds.batch(64), epochs=10)
```

## Model Evaluation
```python
y_pred = m.predict(valid_ds.batch(64))
fpr, tpr, _ = roc_curve(df_valid["label"], y_pred)
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

## Future Enhancements
- Implement ResNet, EfficientNet architectures for improved accuracy.
- Increase dataset diversity for better generalization.
- Deploy as a web-based application for real-time skin cancer detection.



