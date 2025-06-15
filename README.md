# Bank Marketing Campaign Success Prediction & Strategy Optimization

## Project Overview

This project focuses on analyzing the "Bank Marketing" dataset to predict whether a client will subscribe to a term deposit (`deposit` variable) after being contacted during a marketing campaign. Beyond just building a predictive model, a primary goal is to extract actionable insights that can inform and optimize future marketing strategies for the bank, leading to more efficient resource allocation and improved conversion rates.

The dataset contains detailed information about clients, their demographics, financial attributes, and various aspects of the marketing campaigns conducted by a Portuguese banking institution.

## Dataset

* **Source:** [Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data) (Originally from UCI Machine Learning Repository)
* **File Used:** `bank-additional-full.csv` (This dataset is typically preferred for its comprehensiveness).
* **Description:** The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. The classification goal is to predict if the client will subscribe a term deposit (`deposit` column).

### Key Variables:
* **`deposit` (Target Variable):** 'yes' or 'no' - indicates if the client subscribed to a term deposit.
* **`age`:** Client's age.
* **`job`:** Type of job (categorical).
* **`marital`:** Marital status (categorical).
* **`education`:** Client's education level (categorical).
* **`default`:** Has credit in default? (categorical: 'yes', 'no', 'unknown').
* **`balance`:** Average yearly balance, in euros (numerical).
* **`housing`:** Has housing loan? (categorical).
* **`loan`:** Has personal loan? (categorical).
* **`contact`:** Contact communication type (categorical).
* **`day`:** Last contact day of the month (numerical).
* **`month`:** Last contact month of year (categorical).
* **`duration`:** Last contact duration in seconds (numerical). **(IMPORTANT: This feature should be removed for predictive modeling as its value is only known *after* the outcome. However, it's highly valuable for *marketing insights* to guide ongoing call strategy.)**
* **`campaign`:** Number of contacts performed during this campaign for this client (numerical).
* **`pdays`:** Number of days that passed by after the client was last contacted from a previous campaign (numerical; 999 means client was not previously contacted).
* **`previous`:** Number of contacts performed before this campaign and for this client (numerical).
* **`poutcome`:** Outcome of the previous marketing campaign (categorical).
* **Economic Indicators:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` (numerical).

## Project Goal

1.  **Exploratory Data Analysis (EDA):** Gain a deep understanding of the dataset's characteristics, identify patterns, relationships between features, and correlations with the `deposit` target variable.
2.  **Data Preprocessing & Feature Engineering:** Prepare the raw data for machine learning, including handling categorical variables, scaling numerical features, and addressing specific dataset nuances like `pdays` and `duration`.
3.  **Model Building & Evaluation:** Develop and evaluate classification models to predict term deposit subscription, paying special attention to the dataset's class imbalance.
4.  **Derive Actionable Marketing Advice:** Translate model insights and EDA findings into concrete recommendations for the bank's marketing team to improve campaign effectiveness.

## Methodology

### 1. Exploratory Data Analysis (EDA)

* **Initial Inspection:** `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()` to check data types, missing values, and basic statistics.
* **Target Variable Analysis (`deposit`):** Assessed the distribution of 'yes' vs. 'no' subscriptions, revealing a significant class imbalance (high percentage of 'no' deposits).
* **Univariate Analysis:** Explored distributions of individual features (e.g., age distribution, frequency of job types).
* **Bivariate Analysis:**
    * **Categorical vs. `deposit`:** Used cross-tabulations and stacked bar plots to compare subscription rates across different categories (e.g., which `job` types have higher subscription percentages).
    * **Numerical vs. `deposit`:** Employed box plots and overlaid histograms to visualize how numerical features (e.g., `age`, `campaign`) differ between subscribers and non-subscribers. Special attention was paid to `duration` for its insight into call engagement.
* **Multivariate Analysis:** Examined correlations between numerical features using heatmaps to identify potential multicollinearity.

### 2. Data Preprocessing & Feature Engineering

* **Target Encoding:** Mapped the `deposit` variable from ('no', 'yes') to (0, 1).
* **Feature Exclusion:** Excluded `duration` from the predictive features (`X`) due to data leakage, but retained its insights for marketing strategy.
* **`pdays` Transformation:** The `999` value in `pdays` (meaning "not previously contacted") was handled by creating a new binary categorical feature, `pdays_contacted_before`, to differentiate between previously contacted and never-contacted clients.
* **Categorical Encoding:** Applied One-Hot Encoding to all nominal categorical features (e.g., `job`, `marital`, `contact`).
* **Numerical Scaling:** Applied StandardScaler to numerical features to normalize their range, which is beneficial for many machine learning algorithms.
* **Train-Test Split:** Split the data into training and testing sets using `stratify=y` to maintain the class distribution in both sets, crucial for imbalanced data.

### 3. Handling Imbalanced Dataset

Given the high class imbalance, specific strategies were employed to prevent models from being biased towards the majority class:

* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to the training data *within* the pipeline to generate synthetic samples for the minority class (`deposit='yes'`), balancing the training set for the classifiers.
* **Class Weights:** Used `class_weight='balanced'` for Logistic Regression and Random Forest Classifiers, which automatically adjusts weights inversely proportional to class frequencies.
* **`scale_pos_weight` (for XGBoost):** Calculated and applied `scale_pos_weight` to the XGBoost Classifier to give more importance to the minority class.

### 4. Model Selection & Training

A robust `ImbPipeline` (from `imblearn`) was utilized for each model, integrating preprocessing, SMOTE oversampling, and the classifier.

* **Models Implemented:**
    * **Logistic Regression:** As a linear baseline.
    * **Random Forest Classifier:** A powerful ensemble tree-based model.
    * **XGBoost Classifier:** A highly efficient and effective gradient boosting framework.
* **Hyperparameter Tuning:** `RandomizedSearchCV` (with `StratifiedKFold` cross-validation) was performed for each model to find optimal hyperparameters, optimizing for **ROC AUC** (or **F1-score**).

### 5. Model Evaluation

Evaluation focused on metrics crucial for imbalanced datasets, assessing the model's ability to identify the positive class (`deposit='yes'`).

* **Key Metrics:**
    * **Precision:** (TP / (TP + FP)) - How many predicted subscribers were actually subscribers?
    * **Recall:** (TP / (TP + FN)) - How many actual subscribers were correctly identified?
    * **F1-Score:** (2 \* Precision \* Recall) / (Precision + Recall) - Harmonic mean, balancing Precision and Recall.
    * **ROC AUC:** (Area Under ROC Curve) - General discriminative power.
    * **PR AUC:** (Area Under Precision-Recall Curve) - Particularly important for imbalanced data, focusing on positive class performance.
* **Visualization:** Confusion Matrices, ROC Curves, and Precision-Recall Curves were generated for comprehensive understanding.

## Key Findings & Model Performance Summary

The analysis revealed distinct profiles for clients likely to subscribe. The models demonstrated significant improvement over random guessing in identifying these clients.

* **Best Performing Models:** Both **Random Forest Classifier** and **XGBoost Classifier** performed exceptionally well.

* **Key Performance Metrics (for 'Yes Deposit' class - approximately 11% of dataset):**

    | Model                    | ROC AUC | PR AUC | F1-score | Recall | Precision |
    | :----------------------- | :------ | :----- | :------- | :----- | :-------- |
    | **Logistic Regression** | ~0.76   | ~0.76  | ~0.64    | ~0.59  | ~0.71     |
    | **Random Forest** | ~0.77   | ~0.77  | ~0.68    | ~0.63  | ~0.74     |
    | **XGBoost Classifier** | ~0.78   | ~0.78  | ~0.68    | ~0.63  | ~0.74     |

    * **High PR AUC (~0.78):** Indicates strong performance in identifying subscribers effectively while minimizing false positives, crucial for targeted marketing.
    * **Solid F1-score (~0.68):** Shows a good balance between catching subscribers and avoiding wasted efforts.
    * **Good Precision (~0.74):** Means that when the model predicts a subscription, it's correct about 74% of the time, leading to more efficient campaigns.
    * **Decent Recall (~0.63):** The models can identify approximately 63% of all clients who will actually subscribe.

## Actionable Marketing Advice for the Bank

Based on the derived insights and model capabilities, here are concrete recommendations for the bank's marketing strategy:

1.  **Targeted Client Segmentation:**
    * **Prioritize Leads:** Focus outreach on **students, retired individuals, and unemployed clients**. Also, clients in **admin. and technician roles** represent a significant pool of potential subscribers.
    * **Age Focus:** Campaigns should be tailored for **younger adults (20-30s)** and **seniors (60+)**, as these groups exhibit higher responsiveness.
    * **Marital Status:** Consider prioritizing **single clients**.
    * **Education:** Clients with a **university degree or high school education** are more receptive.

2.  **Optimized Contact Strategy:**
    * **Channel Preference:** Strongly favor **`cellular`** contact methods for direct outreach. 'telephone' contact is significantly less effective.
    * **Best Timing:** Concentrate campaign launches and call efforts in **March, October, December, and September**, as these months yield higher subscription rates.
    * **Campaign Frequency Control:** Implement strict limits on contact attempts. A maximum of **3 interactions per client per campaign** is recommended. Excessive contact beyond this threshold (`campaign` count > 3) significantly decreases conversion likelihood and can lead to client annoyance.

3.  **Leverage Past Interactions:**
    * **High-Value Leads:** Clients who had a `poutcome` of **`success`** from a previous marketing campaign are your warmest leads. They should be immediately flagged and prioritized for personalized, high-attention outreach.
    * **Re-engagement Strategy:** For clients with `poutcome` as `failure` or `other`, a different, perhaps softer re-engagement strategy should be considered, or they may be de-prioritized.
    * **Recency:** Explore optimal `pdays` windows for re-contacting clients who were previously contacted but didn't subscribe, as recent past contact can be a positive sign.

4.  **Enhance Agent Training & Call Management:**
    * **Focus on Engagement:** Train marketing agents to identify and nurture engaged conversations. Data shows that **longer calls (`duration`) are highly correlated with successful subscriptions**. Agents should be empowered to extend interactions when client engagement is high, as this indicates a promising lead.

5.  **Strategic Economic Timing:**
    * While not directly controllable, the bank should be aware that campaigns are likely more fruitful during periods of **economic stability or growth** (indicated by lower `emp.var.rate` and `euribor3m`, and higher `cons.conf.idx`). This context can inform broader campaign scheduling.

## How to Use This Repository

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Download the Dataset:**
    * Download `bank-additional-full.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data).
    * Place the `bank-additional-full.csv` file in the root directory of this repository (or update the path in the notebook).
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost
    ```
    *(Note: Ensure your scikit-learn version is 1.3.0 or higher for full compatibility. You may need `pip install --upgrade scikit-learn`.)*
4.  **Run the Jupyter Notebook:**
    Open `your_notebook_name.ipynb` (e.g., `bank_marketing_analysis.ipynb`) in Jupyter Lab/Notebook or a similar environment and run the cells sequentially to reproduce the analysis and model training.

## Requirements

* Python 3.8+
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (version 1.3.0 or higher recommended)
* `imbalanced-learn` (`imblearn`)
* `xgboost`



