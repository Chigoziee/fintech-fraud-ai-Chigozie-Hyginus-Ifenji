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

```
    transaction_type: 4 classes - ['Online' 'POS' 'ATM' 'Transfer']
    device_type: 4 classes - ['Mobile' 'ATM Machine' 'POS Terminal' 'Web']
    location: 5 classes - ['Abuja' 'Lagos' 'Ibadan' 'Kano' 'Port Harcourt']
    time_of_day: 4 classes - ['Afternoon' 'Evening' 'Morning' 'Night']
    day_of_week: 7 classes - ['Fri' 'Tue' 'Wed' 'Sun' 'Thu' 'Mon' 'Sat']
```

- The binary categorical features include:

```
    is_foreign_transaction: 2 classes - [0 1]
    is_high_risk_country: 2 classes - [0 1]
    previous_fraud_flag: 2 classes - [0 1]
```

- customer_id and transaction_id had more than 100 different classes

### Data Preprocessing

- The transaction_time feature was converted into datetime format using pd.to_datetime() function.
- the transaction_time was then used to validate the time_of_day and day_of_week features. This showed that the time_of_day and day_of_week features were not consistent with the transaction_time feature. Therefore, the time_of_day and day_of_week features were modified using the transaction_time feature.

```
    Timestamp: 2024-02-04 22:00:00

    Day of the week from timestamp: Sun
    Day of week from dataFrame: Fri
    Validity: False

    Time of day from timestamp: Night
    Time of day from dataframe: Afternoon
    Validity: False
```

##

## TEMPORAL ANALYSIS RESULTS

Here are the results of the temporal analysis:

- Nights and mornings have the highest risk case of fraud transactions while evenings have the lowest risk case of fraud transactions.

  ![FRAUD COUNT BY TIME OF DAY](utils/images/image-1.png)

- Sundays and thursdays have the highest risk case of fraud transactions while fridays have the lowest risk case of fraud transactions.

  ![FRAUD COUNT BY DAY OF WEEK](utils/images/image.png)

- There were no trends of fraud transactions with respect to weekend or weekdays.
- There were no trends of fraud transactions with respect to hour of the day.

  ![FRAUD COUNT BY HOUR OF DAY](utils/images/image-2.png)

##

## FEATURE ENGINEERING

The following categorical columns were converted to numerical columns using one hot encoding:

- transaction_type
- device_type
- location
  This was because they have 4-5 classes which where unrelated to one another

time_of_day and day_of_week features were converted to numerical columns using cyclical encoding because these features are cyclic in nature and the models need to understand this relationship (i.e monday is close to sunday and far from thursday)

The transaction_amount feature was scaled using RobustScaler because of the skewness of the data. this feature had a mean = 72.595107, std = 73.350547, min = -17.066703, and max = 608.378011.

is_high_risk_country, is_foreign_transaction, previous_fraud_flag, and label_code were not modified as they are binary features.

risk_score was not modified as it is an already scaled numerical feature.

transaction_id, customer_id, and transaction_time were dropped as they are not relevant to the model.

is_high_risk_country and is_foreign_transaction were combined into a new feature called is_high_risk_transaction. This was done because during EDA I notice transactions that is_high_risk_country or is_foreign_transaction are most likely to be fraud. This combination acts like a risk level feature to the model by combining the two models using the logical or (+) operator.

![alt text](utils/images/image-3.png) ![alt text](utils/images/image-4.png)

##

## DATA IMBALANCE HANDLING

The dataset is imbalanced with a fraud rate of 17.1%.

![alt text](utils/images/image-6.png)

This is a common problem in fraud detection. To handle this, SMOTEENN whcih is a combination of SMOTE (Synthetic Minority Over-sampling Technique) and ENN (Edited Nearest Neighbors) was used to oversample the minority class and undersample the majority class.
This method was selected because:

- I want cleaner decision boundaries by removing confusing majority samples.
- I'm dealing with moderate-to-severe class imbalance (17%).
- it works best for this use case case (fraud detection).
- I'm using complex models (XGBoost) that benefit from more refined class separation.

##

## MODELING SELECTION AND TRAINING STRATEGY

The model used for this project is XGBoost. This was chosen because it is a powerful and flexible algorithm that can handle complex relationships between features and the target variable. It also has built-in support for handling imbalanced datasets, which is a common problem in fraud detection. It performs well on tabular data and integrates seamlessly with SHAP for explainability

![alt text](utils/images/image-7.png)

### Training strategy

- The dataset was split into train and test sets with a 80/20 split.
- The model was trained on the train set and evaluated on the test set.
- The model was trained with the following hyperparameters:

```
    n_estimators = 200
    learning_rate = 0.1
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    eval_metric = 'logloss'
```

### MODEL EVALUATION

The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

The model's confusion matrix and metrics are shown below.

```
[[166   0]
 [  0  34]]

TP: 166, FP: 0, TN: 34, FN: 0

Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1-score: 1.0
AUC-ROC: 1.0
```

![alt text](utils/images/image-9.png)

##

## EXPLAINABILITY INSIGHTS

To understand the model's predictions, SHAP (SHapley Additive exPlanations) values were used to quantify the contribution of each feature. The summary plot below reveals the following key insights:

![alt text](utils/images/image-8.png)

### Key Findings

- foreign_or_high_risk is the most influential feature, indicating that transactions from flagged countries or high-risk regions strongly influence fraud predictions.

- transaction_amount and risk_score are equally impactful, suggesting that larger transaction values and elevated risk scores are major signals for fraudulent activity.

- transaction_type_Online has a higher influence compared to POS, implying online transactions carry a higher fraud risk in the dataset.

- Time-based features like day_sin and time_sin show that the model captures temporal fraud patterns—certain hours or days may be more prone to fraudulent activity.

- previous_fraud_flag contributes meaningfully, showing that accounts with a fraud history are more likely to be flagged.

- Location-specific behavior, such as transactions from Port Harcourt, also appears relevant.

- While individually smaller, the collective effect of 14 other features contributes significantly to the model's decisions.

This level of explainability helps validate that the model is aligning with domain knowledge and highlights which features are most actionable for risk teams.

##

## DEPLOYMENT INSTRUCTIONS

**Clone the Repository**

```bash
git https://github.com/Chigoziee/fintech-fraud-ai-Chigozie-Hyginus-Ifenji.git
cd fintech-fraud-ai-Chigozie-Hyginus-Ifenji
```

**Create Virtual Environment**

```bash
 python -m venv .venv
 source .venv/bin/activate  # or .venv\Scripts\activate on Windows powershell
```

**Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Run Streamlit App**

```bash
streamlit run streamlit_app.py
```

The app will be accessible at http://localhost:8501
![alt text](utils/images/image11.png)

The app allows:

- Manual input of transaction data

- Viewing fraud predictions and risk - explanations

- Monitoring weekly fraud trends

![alt text](utils/images/image-10.png)

### **Run FastAPI Server**

```bash
uvicorn api_server:app --reload
```

The FastAPI server will be accessible at http://localhost:8000 for testing and Swagger documentation.

**The predict endpoint**

```bash
http://localhost:8000/predict
```

**Sample payload**

```json
{
  "transaction_amount": 12.33,
  "transaction_type": "Online",
  "device_type": "Web",
  "location": "Port Harcourt",
  "is_foreign_transaction": 1,
  "is_high_risk_country": 0,
  "previous_fraud_flag": 0,
  "risk_score": 1.2332,
  "transaction_time": "2025-07-10 17:00:21"
}
```

**cURL command**

```bash
curl --location 'localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "transaction_amount": 12.33,
    "transaction_type": "Online",
    "device_type": "Web",
    "location": "Port Harcourt",
    "is_foreign_transaction": 1,
    "is_high_risk_country": 0,
    "previous_fraud_flag": 0,
    "risk_score": 1.2332,
    "transaction_time": "2025-07-10 17:00:21"
}'
```

**Response**

```json
{
  "prediction": "1",
  "explanation": {
    "transaction_amount": 1.0228219032287598,
    "is_foreign_transaction": 0.6744084358215332,
    "is_high_risk_country": -0.043741628527641296,
    "previous_fraud_flag": -0.09886821359395981,
    "risk_score": 1.1923465728759766,
    "transaction_type_ATM": 0.0021069394424557686,
    "transaction_type_Online": 0.36210569739341736,
    "transaction_type_POS": -0.24908991158008575,
    "transaction_type_Transfer": -0.006103517487645149,
    "location_Abuja": -0.1782943606376648,
    "location_Ibadan": -0.0013660620898008347,
    "location_Kano": -0.07570569962263107,
    "location_Lagos": -0.026670582592487335,
    "location_Port Harcourt": 0.28420910239219666,
    "device_type_ATM Machine": -0.049883268773555756,
    "device_type_Mobile": -0.01738744229078293,
    "device_type_POS Terminal": -0.04770349711179733,
    "device_type_Web": -0.0006416961550712585,
    "time_sin": -0.17399750649929047,
    "time_cos": -0.09563319385051727,
    "day_sin": 0.6881362795829773,
    "day_cos": -0.17459282279014587,
    "foreign_or_high_risk": 4.545841693878174
  }
}
```

##

## ASSUMPTIONS AND LIMITATIONS

- The system only takes in data consistent with the training data schema.

- The system is limited to static model; retraining requires manual updates.

- Model performance may degrade on unseen patterns or evolving fraud techniques due to lack of continuous monitoring and updating.

- Categorical values outside the fitted encoder’s vocabulary are ignored.
- The weekly fraud trend is based on the data provided.

## FUTURE IMPROVEMENTS

To enhance the system's versitility, performance, and usability, the following improvements are proposed:

### Model Enhancements

- Explore advanced models like LightGBM, CatBoost, or ensemble stacking.

- Add anomaly detection (e.g., Isolation Forest, Autoencoders).

- Implement online/incremental learning for real-time adaptation.

### Data

- Include user behavior metrics (avg transaction size, frequency).

- Automate data ingestion, preprocessing, and retraining.

- Integrate real-time data streams (e.g., Kafka) for live scoring.

### Dashboard/API Improvements

- Add user authentication and secure access.

- Show historical risk trends based on recent transactions and enable batch scoring via CSV.

- Add real-time evaluation using recent transaction data.

- Deploy on cloud platforms for availability and monitoring.
