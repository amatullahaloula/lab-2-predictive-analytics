# Lab 2: Predictive Analytics with Machine Learning

**Course:** CS254 – Introduction to Artificial Intelligence  
**Due Date:** 25th June, 2026  
**Name:** Kasim  
**Institution:** Ashesi University

---

## What this lab is about

This lab covers a full machine learning workflow from start to finish — loading and cleaning real datasets, training models, evaluating them properly, and finding hidden patterns without labels. I worked on two datasets across three different ML tasks: regression, classification, and clustering.

---

## Datasets used

- **NYC Yellow Taxi trips** – used for regression (predicting tip amount)
- **Obesity Level Prediction dataset** – used for both classification (predicting obesity category) and unsupervised clustering

---

## What I did in each section

### Section 1 – Regression (Taxi tips)
Loaded the taxi dataset and explored it to understand the distribution of tip amounts and spot any bad rows (like trips with zero distance). I engineered two new features — `fare_per_mile` and `total_surcharges` — and then split the data 60/20/20 into train, validation, and test sets. I trained a Linear Regression and a Random Forest Regressor and compared their RMSE and R² across all three splits to check for overfitting. The models ended up underfitting rather than overfitting, which I discuss in the reasoning box.

### Section 2 – Classification (Obesity level)
Loaded the obesity dataset, encoded all the categorical columns (binary yes/no columns mapped to 0/1, others one-hot encoded), and did a **stratified** train/validation/test split to keep class proportions balanced across splits. Trained a Random Forest Classifier and evaluated accuracy and macro-F1 on all three splits, plus a confusion matrix on the test set. There was clear overfitting (train accuracy ~99.9% vs test ~93.1%), which I analysed in the reasoning box.

### Section 3 – K-Means Clustering (Obesity, no labels)
Dropped the obesity labels entirely and ran K-Means for k=2 to 10. Used both the Elbow method (inertia) and Silhouette scores to choose k. The silhouette score was highest at k=2 (0.236), suggesting the data naturally groups into two broad weight clusters rather than seven. I visualised the clusters using PCA and compared them against the true labels with a crosstab.

### Section 4 – Reflection
Short written reflection comparing all three tasks — supervised vs unsupervised learning, how regression evaluation differs from classification evaluation, and where overfitting showed up most in this lab.

---

## How to run this notebook

### Option 1 – Google Colab (easiest)
1. Upload `lab_2_predictive_analytics.ipynb` to Colab
2. Upload both CSV files to the Colab session (or put them in your Drive)
3. Click **Runtime → Run all**

### Option 2 – Local Jupyter
```bash
git clone https://github.com/YOUR_USERNAME/lab-2-predictive-analytics.git
cd lab-2-predictive-analytics
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Then open `lab_2_predictive_analytics.ipynb` and run all cells.

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Files in this repo

```
lab-2-predictive-analytics/
├── lab_2_predictive_analytics.ipynb   # main notebook with all code + reasoning
├── yellow_tripdata.csv                # NYC taxi dataset
├── Obesity_level_prediction_dataset.csv  # obesity dataset
├── requirements.txt
└── README.md
```

---

## Key results summary

| Task | Model | Best test metric |
|------|-------|-----------------|
| Regression (tip amount) | Linear Regression | R² = 0.034, RMSE = $5.24 |
| Classification (obesity level) | Random Forest | Accuracy = 93.1%, Macro-F1 = 0.931 |
| Clustering | K-Means (k=2) | Silhouette score = 0.236 |

---

## Notes

- `random_state=42` is used everywhere so results are reproducible
- The scaler is always fitted on training data only to avoid data leakage
- The classification split uses `stratify=y` to keep class balance consistent across splits
