# CreditCardFraud_Detection
 Credit Card Fraud detection using MovieLens dataset and using Isolation Forests using python in Machine Learning.

Here’s a detailed `README.md` description for your credit card fraud detection project on GitHub:

---

# Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains transactions made by credit card holders, where the objective is to distinguish between legitimate (normal) and fraudulent transactions. Given the highly imbalanced nature of the dataset, we employ anomaly detection methods to identify fraudulent transactions.

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection** dataset, which is available on Kaggle. It contains credit card transactions made in September 2013 by European cardholders. The dataset consists of 284,807 transactions, out of which 492 are fraudulent, making the dataset highly imbalanced.

- **Features**: The dataset contains 30 features:
  - `V1, V2, ..., V28`: Principal Component Analysis (PCA) transformed features to maintain anonymity.
  - `Amount`: Transaction amount.
  - `Time`: Seconds elapsed between the first transaction and this transaction.
  - `Class`: The label where 0 represents normal transactions and 1 represents fraudulent transactions.

## Methodology

### 1. **Exploratory Data Analysis**
   - Visualized the distribution of normal and fraudulent transactions.
   - Observed key statistics for fraudulent transactions, particularly focusing on transaction amounts.

### 2. **Modeling Techniques**
   To address the imbalanced dataset, we used **anomaly detection models** that are specifically designed to handle imbalanced data. These models include:
   
   - **Isolation Forest**: A tree-based model that isolates anomalies by randomly selecting features and splitting data points.
   - **Local Outlier Factor (LOF)**: A model that measures the local density of data points, where outliers have much lower densities compared to their neighbors.
   - **One-Class SVM**: A Support Vector Machine algorithm designed for anomaly detection.

   In this project, the **Isolation Forest** model was implemented to detect fraudulent transactions.

### 3. **Performance Metrics**
   The following metrics were used to evaluate model performance:
   - **Accuracy**: Measures the overall correctness of the model.
   - **Classification Report**: Provides precision, recall, F1-score, and support for both classes.

## Results

- **Fraud Rate**: The dataset has a fraud rate of approximately `0.0017`, with 492 fraudulent transactions out of 284,807 total transactions.
- **Model Performance**:
  - Number of misclassified transactions: `<insert n_errors>`
  - Accuracy: `<insert accuracy_score>`
  - Precision, recall, F1-score for each class can be found in the classification report.

## How to Use

### Prerequisites

Install the necessary Python packages by running:

```bash
pip install -r requirements.txt
```

### Running the Model

1. Download the dataset and save it as `creditcard.csv`.
2. Run the script:

```bash
python CreditCard_FraudDetection.py
```

### Files in the Repository

- **fraud_detection.py**: Main script containing the code for data preprocessing, modeling, and evaluation.
- **creditcard.csv**: Dataset used for training and evaluation (ensure this file is present in the root directory).
- **requirements.txt**: Contains the required dependencies for the project.

## Conclusion

The project demonstrates the use of anomaly detection techniques, particularly Isolation Forest, to effectively detect fraudulent credit card transactions. Despite the class imbalance, the model is able to identify fraudulent transactions with reasonable accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Author : Atchuth V
