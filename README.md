# 🛡️ Fraud Detection System

Welcome to the **Fraud Detection System** project! This project aims to develop a robust machine learning model to detect fraudulent transactions using a comprehensive dataset of financial transactions. The goal is to identify suspicious activities and reduce fraud in digital payments through advanced data analysis and machine learning techniques.

## 📋 Project Overview

In today's digital world, fraud detection is a critical challenge for financial institutions. This project leverages supervised learning algorithms to detect fraud in financial transactions with a focus on accuracy and interpretability. We tackle the problem of imbalanced data using techniques like **SMOTE**, and implement robust pre-processing and outlier detection methods.

## 🛠️ Technologies Used

- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Models**: Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN)
- **Data Handling**: SMOTE for imbalanced data, feature scaling, and outlier removal
- **Deployment**: Docker (optional) for containerization

## 📊 Dataset

The dataset contains **1 million transactions** with the following features:

- `distance_from_home`: Distance between the user's home and the transaction location
- `distance_from_last_transaction`: Distance from the previous transaction
- `ratio_to_median_purchase_price`: Ratio of the transaction amount to the median purchase price
- `repeat_retailer`: Indicates if the transaction was made with a repeat retailer (1 = yes, 0 = no)
- `used_chip`: Indicates if the transaction used a chip card (1 = yes, 0 = no)
- `used_pin_number`: Indicates if a PIN was used (1 = yes, 0 = no)
- `online_order`: Indicates if the transaction was made online (1 = yes, 0 = no)
- `fraud`: Target variable indicating if the transaction is fraudulent (1 = yes, 0 = no)

The dataset is **imbalanced**, with only 9% of transactions being fraudulent.

## 🧪 Methodology

1. **Data Exploration**:
   - Analyzed the dataset for missing values (none found).
   - Detected and removed outliers to ensure clean data.
   - Applied SMOTE to handle the class imbalance problem.

2. **Feature Engineering**:
   - Created new features based on domain knowledge.
   - Scaled the features using standardization.

3. **Model Implementation**:
   - Tested three models: Logistic Regression, SVM, and KNN.
   - Evaluated the models using metrics such as Accuracy, Precision, Recall, and F1-Score.

4. **Evaluation**:
   - Addressed overfitting with regularization techniques.
   - Selected the best model based on a balance between precision and recall.
