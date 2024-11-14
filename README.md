

# Credit Card Approval Prediction with Machine Learning

This project builds an automatic credit card approval predictor using machine learning, which mirrors real-world processes used by commercial banks to evaluate credit applications. The model automates the analysis of applications, helping reduce manual workload, improve accuracy, and save time.

## Dataset
The dataset used in this project is a subset of the **Credit Card Approval dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). Each row represents a credit card application with various attributes and a target label indicating whether the application was approved.

## Libraries Used
- **Pandas**: Data manipulation and analysis
- **NumPy**: Array and numerical operations
- **Scikit-learn**: Machine learning models, data preprocessing, and evaluation tools

## Project Workflow

1. **Data Loading and Exploration**
   - Load the data using `pandas`.
   - Display a sample to understand data structure and types.

2. **Data Preprocessing**
   - **Missing Value Handling**: Replaces missing values (`?`) with `NaN`.
   - **Imputation**: Fills missing values with the most frequent value for categorical data and the mean for numerical data.
   - **Encoding**: Converts categorical variables into dummy/indicator variables.

3. **Feature and Target Extraction**
   - Separates the features (X) from the target variable (y), which indicates application approval.

4. **Data Splitting**
   - Divides data into training and testing sets using `train_test_split` with a 67-33 split.

5. **Data Scaling**
   - Standardizes the feature variables using `StandardScaler` to improve model performance.

6. **Model Training and Evaluation**
   - Trains a **Logistic Regression** model on the scaled training data.
   - Evaluates model performance using a **Confusion Matrix**.

7. **Hyperparameter Tuning**
   - Uses `GridSearchCV` for hyperparameter tuning to optimize `tol` and `max_iter` parameters.
   - Evaluates model performance with cross-validation.

8. **Testing the Optimized Model**
   - Extracts the best model from GridSearch results and tests it on the test dataset.
   - Prints the model's accuracy score on the test set.

## Results
The best-tuned model achieved:
- **Training Accuracy**: ~81.8% with the parameters `{'max_iter': 100, 'tol': 0.01}`
- **Test Accuracy**: ~79.4%

### Confusion Matrix (Training Set)
```
[[203   1]
 [  1 257]]
```

## Conclusion
This project demonstrates an effective approach for automating credit card application approvals. Logistic Regression, along with proper data preprocessing and hyperparameter tuning, can serve as a solid baseline model in predicting credit approvals.

## Requirements
Install the following libraries:
```bash
pip install pandas numpy scikit-learn
```

## How to Run
1. Load the dataset and run the code sequentially.
2. Follow along with each section to see data transformations, model training, and evaluation steps.

## License
This project is open source and free to use under the [MIT License](LICENSE).

--- 

Feel free to modify and expand as needed.
