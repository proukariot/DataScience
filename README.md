# Loan Default Prediction Project - README
## Project Overview
This project aims to predict the loan status (loan_status) as either fully paid or default using various classification models. The dataset contains various features related to loan applications, credit history, employment, income, and more. Six classification models were applied: BaggingClassifier, XGBoost, LogisticRegression, AdaBoostClassifier, GradientBoostingClassifier, and RandomForestClassifier. Additionally, Principal Component Analysis (PCA) was used for dimensionality reduction, and hyperparameter tuning was performed to optimize the models' performance.

## Dataset Description
#### The dataset contains the following columns: 
['loan_amnt', 'credit_month', 'int_rate', 'emp_length', 'annual_inc', 'verification_status', 'loan_status', 'dti', 'delinq_2yrs',
       'fico_range', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc',
       'pub_rec_bankruptcies', 'grade_numeric', 'years_of_credit_history',
       'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN',
       'home_ownership_RENT', 'purpose_credit_card',
       'purpose_debt_consolidation', 'purpose_home_improvement',
       'purpose_major_purchase', 'purpose_other']
## Project Steps
### 1. Data Preprocessing
Handling Missing Values: Missing values in the dataset were identified and handled appropriately.
Feature Scaling: The numerical features were scaled to ensure that they are on a similar scale. This helps improve the performance of machine learning models.
Feature Engineering: Additional features, such as years_of_credit_history, were created to enhance the predictive power of the models.
### 2. Principal Component Analysis (PCA)
Dimensionality Reduction: PCA was applied to reduce the dimensionality of the dataset while retaining as much variance as possible. This step helps in reducing model complexity and computation time.
Variance Retained: The number of components was chosen to retain a significant amount of variance (e.g., 95%) from the original features.
### 3. Model Selection
#### Models Used:
BaggingClassifier: An ensemble method that fits multiple instances of a base estimator on different random subsets of the dataset and averages their predictions.

XGBoost: An efficient and scalable implementation of gradient boosting, widely used for its high performance.

LogisticRegression: A simple, interpretable model that predicts probabilities using a logistic function.

AdaBoostClassifier: An ensemble method that combines multiple weak learners to create a strong classifier.

GradientBoostingClassifier: A powerful ensemble technique that builds models sequentially and reduces the bias of the combined model.

RandomForestClassifier: An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

### 4. Hyperparameter Tuning
Grid Search: Performed grid search to identify the best combination of hyperparameters for each model. This involved defining a grid of hyperparameters and searching exhaustively through this space.

Cross-Validation: Used k-fold cross-validation to ensure the model's robustness and generalization capability.
### 5. Model Evaluation
#### Metrics Used:

Accuracy: The proportion of correctly classified instances.

Precision, Recall, F1-Score: Additional metrics to evaluate the model's performance, particularly in handling imbalanced data.

ROC-AUC: Used to measure the model's ability to distinguish between classes.

Comparison of Models: The performance of all six models was compared based on the above metrics. The best-performing model was selected for the final prediction.

### 6. Final Model Selection
Based on the evaluation metrics, the model with the best performance on the test data was selected as the final model for deployment.
### Conclusion
The project successfully developed a predictive model for loan status using a variety of machine learning techniques. By applying PCA and hyperparameter tuning, the model's performance was optimized, leading to a more robust and accurate prediction.

### How to Run
Clone the repository.
Install the necessary dependencies and libraries. Run the notebook or script file to preprocess the data, train the models, and evaluate their performance.
### Acknowledgements
The dataset used in this project was sourced from Kaggle. Thanks to the developers and maintainers of the libraries used in this project.
