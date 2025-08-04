# FRAUD DETECTION IN FINTECH PLATFORMS

## PROJECT OVERVIEW

This project is a Fraud Detection System designed to identify potentially fraudulent financial transactions in real-time using machine learning. Built with a production-grade architecture, the system offers:

- An XGBoost model trained on transactional features.

- Feature engineering with scaling, encoding, and time-based transformations.

- SHAP-based explanations for model predictions.

- A FastAPI backend that returns fraud risk and interpretability info.

- A Streamlit dashboard for manual input, predictions, and visual insights.

The system is built for financial institutions and fintechs to detect and understand fraud effectively.

##

## DATA UNDERSTANDING AND PROCESSING

EDA was carried out on the data and these are the following insights:

- Dataset had 1000 entries
- Dataset had 13 features: transaction_id, customer_id, transaction_amount,transaction_type, device_type, location, time_of_day, day_of_week, is_foreign_transaction, is_high_risk_country, previous_fraud_flag, transaction_time, risk_score
- Dataset had 1 target variable: label_code which is a binary classification target
- Dataset had no null values
- Dataset had no duplicate values
- Dataset had fraud cases: 171 and non-fraud cases: 829
- Database has 2 numerical features, 1 datetime feature and 10 categorical features
- The non binary categorical features include:
  - transaction_type: 4 classes - ['Online' 'POS' 'ATM' 'Transfer']
  - device_type: 4 classes - ['Mobile' 'ATM Machine' 'POS Terminal' 'Web']
  - location: 5 classes - ['Abuja' 'Lagos' 'Ibadan' 'Kano' 'Port Harcourt']
  - time_of_day: 4 classes - ['Afternoon' 'Evening' 'Morning' 'Night']
  - day_of_week: 7 classes - ['Fri' 'Tue' 'Wed' 'Sun' 'Thu' 'Mon' 'Sat']
- The binary categorical features include:
  - is_foreign_transaction: 2 classes - [0 1]
  - is_high_risk_country: 2 classes - [0 1]
  - previous_fraud_flag: 2 classes - [0 1]
- customer_id and transaction_id are unique identifiers for each transaction with > 100 different classes

### Data Preprocessing

- The transaction_time feature was converted into datetime format using pd.to_datetime() function.
- the transaction_time was then used to validate the time_of_day and day_of_week features. This showed that the time_of_day and day_of_week features were not consistent with the transaction_time feature. Therefore, the time_of_day and day_of_week features were modified using the transaction_time feature.

##

## FEATURE ENGINEERING

The following categorical columns were converted to numerical columns using one hot encoding:

- transaction_type
- device_type
- location
  this was because they have 4-5 classes which where unrealed to one another

time_of_day and day_of_week features were converted to numerical columns using cyclical encoding because these features are cyclic in nature and the models need to understand this relationship (i.e monday is close to sunday and far from thursday)

The transaction_amount featurewas scaled using RobustScaler beacsue of the skewness of the data. this feature had a mean = 72.595107, std = 73.350547, and max = 608.378011.

is_high_risk_country, is_foreign_transaction, previous_fraud_flag, and label_code were not modified as they are binary features.

risk_score was not modified as it is an already scaled numerical feature.

transaction_id, customer_id, and transaction_time were dropped as they are not relevant to the model.

is_high_risk_country and is_foreign_transaction were combined into a single feature called is_high_risk_transaction. This was done because during EDA I notice transactions that is_high_risk_country and is_foreign_transaction are most likely to be fraud. so this combination acts like a risk level feature to the model by combining the two models using the logical or (+) operator.

##

## MODELING SELECTION AND TRAINING STRATEGY

The model used for this project is XGBoost. This was chosen because it is a powerful and flexible algorithm that can handle complex relationships between features and the target variable. It also has built-in support for handling imbalanced datasets, which is a common problem in fraud detection. It has high performance on tabular data, Compatibility with SHAP for explainability and Compatibility with SHAP for explainability.

### Training strategy

- The dataset was split into train and test sets with a 80/20 split.
- The model was trained on the train set and evaluated on the test set.
- The model was trained with the following hyperparameters:
  - n_estimators = 200
  - learning_rate = 0.1
  - max_depth = 6
  - subsample = 0.8
  - colsample_bytree = 0.8
  - eval_metric = 'logloss'
