# ANN Fetal Health Classification

## Overview

This project applies an **Artificial Neural Network (ANN)** to classify fetal health based on several health indicators. The dataset used includes various features that track fetal health conditions, such as fetal movements, accelerations, decelerations, uterine contractions, and other vital measurements. The goal of this model is to accurately predict the health status of a fetus (normal, suspected, or pathological) based on these inputs.

## Dataset Features

The dataset contains various fetal health indicators, including but not limited to:
- **Baseline Value**
- **Accelerations**
- **Fetal Movement**
- **Uterine Contractions**
- **Decelerations (Light, Severe, Prolonged)**
- **Short-Term Variability (Mean, Abnormal)**
- **Long-Term Variability**
- **Histogram Data** (Width, Min, Max, Peaks, Variance, etc.)

## Model

The classification task is performed using an **Artificial Neural Network (ANN)** with the following structure:
- **Input Layer**: Receives all feature variables (e.g., accelerations, uterine contractions, etc.)
- **Hidden Layers**: Two or more hidden layers with ReLU activation to capture complex relationships between input features.
- **Output Layer**: A softmax layer to classify the fetal health into three categories: 
  - Normal
  - Suspected
  - Pathological

## Objective

The main objectives of this project are:
- **Predict Fetal Health Status**: Classify the health status as normal, suspected, or pathological based on given features.
- **Evaluate Model Performance**: Use metrics such as accuracy, precision, recall, and F1-score to assess the model's performance.

## Tools & Libraries

- **TensorFlow / Keras**: For building and training the ANN model.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For splitting the data, scaling, and evaluation.
- **Matplotlib / Seaborn**: For visualizing the data and model performance.

## Potential Use Cases

This project can be beneficial for:
- **Medical Diagnosis**: Assisting healthcare professionals in identifying potential risks in fetal health during pregnancy.
- **Research in Healthcare AI**: Exploring how AI can contribute to fetal health monitoring and predictive diagnostics.

## How to Run

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/fetal-health-classification.git
    cd fetal-health-classification
    ```

2. **Install the Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Model**:
    You can load the dataset, preprocess it, and then train the ANN model with the following script:
    ```python
    python train_model.py
    ```

4. **Evaluate the Model**:
    After training, evaluate the model's accuracy, precision, and recall on the test data.

## Contributions

We welcome contributions to this project! If you have suggestions or improvements, feel free to submit a pull request or open an issue.


This project applies an **Artificial Neural Network (ANN)** 

You can view the implementation of this model on Kaggle:  
[ANN Fetal Health Classification on Kaggle](https://www.kaggle.com/code/elmuatazelhariri/ann-fetal-health-classification/edit/run/199488655)
