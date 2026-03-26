# 📊 Advanced Analytics Project

A end-to-end machine learning project covering regression, classification, clustering, and time series analysis using two real-world datasets.

---

## 📁 Project Structure

```
├── LucaFrittitta_AdvancedAnalytics1.ipynb   # Main notebook
├── supermarket_sales.csv                     # Dataset 1 — Supermarket transactions
└── apple_quality.csv                         # Dataset 2 — Apple quality classification
```

---

## 📦 Datasets

### 1. Supermarket Sales
Contains transactional data from a supermarket chain, including product lines, payment methods, customer types, and financial metrics. Used for **regression** and **time series** tasks.

**Target variable:** `Rating` — represents how profitable each transaction was for the supermarket.

### 2. Apple Quality
Contains physical measurements of apples (size, weight, sweetness, etc.) with a binary quality label (`good` / `bad`). Used for **classification** and **clustering** tasks.

**Target variable:** `Quality` — binary label indicating apple quality.

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis
- Computed mean, median, mode, and standard deviation of the target variable using NumPy
- Visualized the distribution of `Rating` (uniform) and `gross income` (right-skewed)
- Measured skewness of `gross income` using pandas `.skew()`

### 2. Data Preprocessing
- Dropped irrelevant columns: `Invoice ID`, `Tax 5%`, `Total`, `Date`, `Time`, `cogs`, `gross margin percentage`
- Applied **One-Hot Encoding** to categorical variables: `Branch`, `City`, `Customer type`, `Gender`, `Product line`, `Payment`
- Applied **StandardScaler** to numerical features: `Unit price`, `gross income`
- Handled missing values in the apple dataset with `.dropna()`
- Encoded binary target variable (`good` → 1, `bad` → 0)

### 3. Regression — Predicting Supermarket Rating

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline model, evaluated with MSE and MAE |
| **Polynomial Regression** (degree=2) | Improved performance over linear baseline |

**Metrics used:** Mean Squared Error (MSE), Mean Absolute Error (MAE)

### 4. Classification — Apple Quality Prediction

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline classifier, evaluated before and after scaling |
| **Decision Tree** (Gini) | Improved performance over Logistic Regression |

**Metrics used:** F1 Score, Precision, Recall, Confusion Matrix  
**Target threshold:** F1 > 0.80

Feature importance was extracted from the Decision Tree to identify the most influential variables.

### 5. Clustering — Unsupervised Apple Segmentation

| Configuration | Result |
|---|---|
| `KMeans(n_clusters=2)` | Two clusters: good / bad |
| `KMeans(n_clusters=3)` | Three clusters: good / medium / bad |

Labels are unsupervised — no ground truth used.

### 6. Time Series — Gross Income Over Time

- Extracted time component from supermarket transactions
- Converted time to minutes for numerical modeling
- Applied **Linear Regression** to model gross income trend over time
- Applied **XGBoost Regressor** (optional) as a more powerful non-linear alternative

**Metrics used:** MSE, MAE

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Statistical Analysis | `scipy` |

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/advanced-analytics-project.git
cd advanced-analytics-project
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```

3. Update the dataset paths in the notebook:
```python
# Replace local paths with your own
regression_raw_dataset = pd.read_csv("supermarket_sales.csv")
classification_dataset = pd.read_csv("apple_quality.csv")
```

4. Run the notebook:
```bash
jupyter notebook LucaFrittitta_AdvancedAnalytics1.ipynb
```

---

## 📈 Results Summary

| Task | Model | Best Metric |
|---|---|---|
| Regression | Polynomial Regression (deg=2) | Lower MSE/MAE vs Linear |
| Classification | Decision Tree | F1 > 0.80 |
| Clustering | KMeans (k=2, k=3) | Unsupervised segmentation |
| Time Series | XGBoost (optional) | Lower MSE/MAE vs Linear |

---

## 👤 Author

**Luca Frittitta**  
Data Science & Advanced Analytics — Start2Impact University  
📧 lucafrittitta@gmail.com
