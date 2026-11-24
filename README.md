# Exp No. 2: Exploratory Data Analysis (EDA)

## Aim

To perform Exploratory Data Analysis (EDA) on the heart disease dataset to understand the data distribution, relationships, and key patterns useful for business insights and model building.

## Dataset

The dataset used for this experiment is `heartdisease.csv`. It contains medical and demographic information of 250 patients across 17 attributes, such as `Age`, `Gender`, cholesterol levels (`Totalcholestrol`), `BMI`, blood pressure (`SysBP`, `DiaBP`), and the target variable (`TenYearCHD`) indicating whether the patient developed coronary heart disease within ten years.

## Algorithm

1. Import required Python libraries (`numpy`, `pandas`, `seaborn`, `matplotlib`).
2. Load the dataset into a Pandas DataFrame.
3. Display the dataset structure, shape, and descriptive statistics.
4. Check for missing values and duplicate rows.
5. Examine correlations among numeric features using a correlation matrix and heatmap.
6. Visualize univariate distributions using histograms, boxplots, and countplots.
7. Analyze bivariate and multivariate relationships using scatter plots and pairplots.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Basic Exploration** | `df.info()` | Inspect dataset structure and data types |
| | `df.describe()` | Generate descriptive statistics |
| | `df.isnull().sum()` | Detect missing values in each column |
| | `df.duplicated().sum()` | Identify duplicate rows |
| | `df.corr()` | Find correlations between numeric variables |
| **Univariate Analysis** | `sns.histplot()` | Plot histogram (distribution of single variable) |
| | `sns.boxplot()` | Identify outliers using boxplot |
| | `sns.countplot()` | Count occurrences of categorical variables |
| **Bivariate Analysis** | `sns.scatterplot()` | Display relationship between two variables |
| | `sns.heatmap()` | Show correlation matrix visually |
| **Multivariate Analysis** | `sns.pairplot()` | Show pairwise relationships across variables |
| **Visualization Controls** | `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display plotted charts |

## Program & Output

### Program

```python
# Exploratory Data Analysis (EDA) on the Heart Disease Dataset

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
sns.set_style("whitegrid")

# Step 2: Load the dataset
df = pd.read_csv("heartdisease.csv")

# Step 3: Structure, shape and descriptive statistics
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())

# Step 4: Missing values and duplicates
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Step 5: Correlation matrix and heatmap
numeric_df = df.drop(columns=["S. No"]).select_dtypes(include=np.number)
corr = numeric_df.corr()
print("\nCorrelation of features with TenYearCHD:")
print(corr["TenYearCHD"].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Heart Disease Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=120)
plt.show()

# Step 6: Univariate analysis
num_cols = ["Age", "BMI", "Totalcholestrol", "SysBP"]

# 6a. Histograms
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, bins=20, color="steelblue", ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/histograms.png", dpi=120)
plt.show()

# 6b. Boxplots (outlier detection)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(y=df[col], color="lightcoral", ax=ax)
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig("images/boxplots.png", dpi=120)
plt.show()

# 6c. Countplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
sns.countplot(x="Gender", data=df, hue="Gender", palette="Set2", legend=False, ax=axes[0])
axes[0].set_title("Patients by Gender")
sns.countplot(x="currentsmoker", data=df, hue="currentsmoker", palette="Set1", legend=False, ax=axes[1])
axes[1].set_title("Current Smoker (0 = No, 1 = Yes)")
sns.countplot(x="TenYearCHD", data=df, hue="TenYearCHD", palette="Dark2", legend=False, ax=axes[2])
axes[2].set_title("TenYearCHD (0 = No, 1 = Yes)")
for ax in axes:
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("images/countplots.png", dpi=120)
plt.show()

print("\nTenYearCHD class distribution:")
print(df["TenYearCHD"].value_counts())
print(df["TenYearCHD"].value_counts(normalize=True).round(3))

# Step 7: Bivariate and multivariate analysis
# 7a. Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x="SysBP", y="DiaBP", hue="TenYearCHD", data=df, palette="coolwarm", ax=axes[0])
axes[0].set_title("Systolic vs Diastolic Blood Pressure")
sns.scatterplot(x="Age", y="Totalcholestrol", hue="TenYearCHD", data=df, palette="coolwarm", ax=axes[1])
axes[1].set_title("Age vs Total Cholesterol")
plt.tight_layout()
plt.savefig("images/scatterplots.png", dpi=120)
plt.show()

# 7b. Pairplot
pair = sns.pairplot(df[["Age", "BMI", "SysBP", "Totalcholestrol", "TenYearCHD"]],
                    hue="TenYearCHD", palette="coolwarm", corner=True)
pair.figure.suptitle("Pairplot of Key Features", y=1.02, fontsize=14, fontweight="bold")
pair.savefig("images/pairplot.png", dpi=100)
plt.show()
```

### Output

**1. Dataset overview, missing values and duplicates**

```
Shape of dataset: (250, 17)

Missing values per column:
All 17 columns: 0

Duplicate rows: 0
```

**2. Correlation with the target variable (TenYearCHD)**

```
Prevalentstroke    0.127883
Diabetes           0.126330
SysBP              0.114203
PrevalentHyp       0.052606
Age                0.034526
BPMeds             0.034323
DiaBP              0.028685
BMI                0.026675
Glucose           -0.015735
Heartrate         -0.044918
Totalcholestrol   -0.078204
currentsmoker     -0.082408
Education         -0.086376
Cigsperday        -0.096773
```

**3. Correlation heatmap**

<img width="1440" height="1080" alt="image" src="https://github.com/user-attachments/assets/ff005d63-e1b9-4e75-8bba-195b3abfd3f8" />


**4. Histograms (Age, BMI, Totalcholestrol, SysBP)**

<img width="1320" height="960" alt="image" src="https://github.com/user-attachments/assets/61dbfb59-a4fd-4d9e-b1c9-f492c505538f" />


**5. Boxplots (outlier detection)**

<img width="1320" height="960" alt="image" src="https://github.com/user-attachments/assets/3a389ea5-45ee-4694-aa43-5e99a2c53e3f" />


**6. Countplots (Gender, Smoker, TenYearCHD)**

<img width="1680" height="540" alt="image" src="https://github.com/user-attachments/assets/e8e94c0a-abcf-4922-9995-d4ab1991675e" />


```
TenYearCHD class distribution:
0    204   (81.6%)
1     46   (18.4%)
```

**7. Scatter plots**

<img width="1680" height="600" alt="image" src="https://github.com/user-attachments/assets/550c200f-1a20-4055-bd69-8f0ccf14bb9d" />


**8. Pairplot**

<img width="1090" height="1023" alt="image" src="https://github.com/user-attachments/assets/6b288db9-0c48-48af-9849-82770ee49de2" />


### Key Observations

* The dataset is clean: 250 records, 17 attributes, no missing values and no duplicate rows.
* The target is imbalanced: 81.6% of patients have no CHD (204) and 18.4% have CHD (46). A model trained on this data would need class balancing or metrics beyond accuracy.
* `Age`, `BMI`, `Totalcholestrol` and `SysBP` are roughly symmetric, with skewness between -0.27 and 0.27. The boxplots show very few outliers (0 for Age, 1 for BMI, 2 for Totalcholestrol, 1 for SysBP).
* The only strong correlation is between `currentsmoker` and `Cigsperday` (0.78), which is expected, since non-smokers smoke zero cigarettes.
* Correlations with `TenYearCHD` are all weak (below 0.13). The largest are `Prevalentstroke` (0.13), `Diabetes` (0.13) and `SysBP` (0.11).
* Patients with diabetes had a CHD rate of 35% compared with 17% for those without. The 6 patients with a prior stroke had a 50% rate, but that group is too small to be reliable.
* The pairplot shows no clear separation between CHD and non-CHD patients on any pair of features.

## Result

Exploratory Data Analysis was successfully performed on the heart disease dataset of 250 records. The data has no missing values or duplicates, and the key numeric features are roughly symmetric with few outliers. The target variable is imbalanced (81.6% vs 18.4%), and no single feature correlates strongly with `TenYearCHD`; the strongest relationships are with prior stroke, diabetes and systolic blood pressure. These findings suggest that predicting heart disease from this dataset needs several features combined and attention to class imbalance.
