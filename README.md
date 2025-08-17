# 🖼️ Image Classification using CNN & MobileNet

This project focuses on building an **image classification model** using **Convolutional Neural Networks (CNNs)** and **transfer learning with MobileNet**. The model classifies images into predefined categories after training on a dataset.

---

## 📂 Dataset

The dataset organized in the following structure:

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...
```

Each subfolder should contain images belonging to that class.

*(Update this section with your actual dataset link if it’s from Kaggle or another source.)*

---

## 🚀 Project Workflow

1. **Data Preprocessing**

   * Loaded images using OpenCV.
   * Resized all images to **32×32** pixels.
   * Normalized pixel values (0–1).
   * Applied data augmentation with `ImageDataGenerator`.

2. **Model Architecture**

   * Implemented CNN with Conv2D, MaxPooling, Flatten, Dense, Dropout layers.
   * Also experimented with **MobileNet** (transfer learning).

3. **Training**

   * Optimizer: **Adam**
   * Loss function: **categorical\_crossentropy**
   * Metrics: **accuracy**

4. **Evaluation**

   * Used **confusion matrix** and **accuracy score**.
   * Plotted training/validation accuracy and loss.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **Scikit-learn**
* **Matplotlib**
* **Tabulate**

---

## 📊 Results

* Achieved high accuracy on the test set.
* Confusion matrix showed strong performance across most classes.
* Transfer learning with **MobileNet** improved accuracy compared to a custom CNN.

---

## 📦 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/image-classification-cnn.git
   cd image-classification-cnn
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset:

   * Place dataset inside a `data/` folder following the structure above.

4. Run the notebook:

   ```bash
   jupyter notebook code.ipynb
   ```

---

## 🔮 Future Improvements

* Train on higher resolution images (64×64, 128×128).
* Try other pretrained models (ResNet, EfficientNet).
* Deploy model as a web app (Streamlit/Flask).

---

## ✨ Acknowledgements

* TensorFlow/Keras for deep learning models.
* MobileNet architecture for transfer learning.
* OpenCV for image preprocessing.

