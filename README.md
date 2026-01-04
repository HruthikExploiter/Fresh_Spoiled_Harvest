# 🍎 FreshHarvest AI -- Fresh vs Spoiled Fruit Detection

FreshHarvest AI is a deep learning--powered web application that
classifies fruits and vegetables as **Fresh** or **Spoiled** using a
Convolutional Neural Network (CNN).
The model is trained on thousands of real fruit images and optimized
using **Optuna** for high accuracy and generalization.

## 🚀 Live Demo

([https://freshspoiledharvest.streamlit.app/])

## 📌 Features

-   Upload an image or provide an image URL
-   Detects whether produce is **Fresh** or **Spoiled**
-   Optimized CNN trained using **hyperparameter tuning (Optuna)**
-   User-friendly **Streamlit** web interface
-   Supports Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo,
    and Tomato

## 🧠 How It Works

1.  User uploads a fruit or vegetable image
2.  The image is preprocessed and resized
3.  A trained CNN predicts freshness
4.  The result (Fresh / Spoiled) is displayed with confidence

## 🏗 Model Architecture

The CNN consists of: - 3 Convolutional layers
- ReLU activations
- MaxPooling
- Dropout for regularization
- Fully connected layers for classification

The model was optimized using **Optuna** and achieved **\~93% accuracy**
on validation data.

## 📂 Project Structure

    fresh_spoiled_harvest/
    ├── app/
    │   └── app.py
    ├── artifacts/
    │   └── fresh_spoiled_optuna_cnn.pth
    ├── requirements.txt
    └── README.md

## ▶ How to Run Locally

``` bash
git clone https://github.com/HruthikExploiter/Fresh_Spoiled_Harvest.git
cd fresh_spoiled_harvest
pip install -r requirements.txt
streamlit run app/app.py
```

## ⚠ Important Note

This model is trained only on: Banana, Lemon, Lulo, Mango, Orange,
Strawberry, Tamarillo, and Tomato.\
Predictions for other fruits or vegetables may be less accurate.

## 📊 Performance

Validation Accuracy: \~93%

## 👨‍💻 Author

Hruthik Gajjala
