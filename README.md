# Titanic Dataset Exploratory Data Analysis (EDA)

This project contains an Exploratory Data Analysis (EDA) on the classic [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv). The notebook covers initial data loading, inspection, handling missing values, and data preprocessing steps.

## 📊 Dataset Overview

The dataset consists of passenger information from the Titanic, filtered to focus on key features:
- **Total Records:** 891 rows
- **Selected Columns:** 5 columns (`Name`, `Sex`, `Age`, `Fare`, `Survived`)

### Columns & Data Types:
* **Name** (`object`): Passenger's full name.
* **Sex** (`object` / encoded to `int64`): Passenger's gender (mapped to numeric format).
* **Age** (`float64`): Passenger's age (contains 177 missing values).
* **Fare** (`float64`): Ticket price paid by the passenger.
* **Survived** (`int64`): Survival status (0 = No, 1 = Yes).

---

## 🛠️ Code & Implementation Steps

### 1. Importing Libraries & Loading Data
The analysis begins by loading essential Python data science libraries and reading the dataset directly from the source repository:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

warnings.filterwarnings("ignore")
%matplotlib inline

# Load the Titanic dataset
url = "[https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)"
df = pd.read_csv(url)
df = df[['Name', 'Sex', 'Age', 'Fare', 'Survived']]

```

### 2. Initial Data Inspection

* Checked dimensions using `df.shape` (891 rows, 5 columns).
* Inspected data types and non-null counts using `df.info()`.
* Generated statistical summaries using `df.describe()`.
* Identified missing values (`Age` has 177 missing entries) using `df.isnull().sum()`.

### 3. Data Preprocessing & Mapping

Categorical text values in the `Sex` column were mapped to numerical values (`male` = 1, `female` = 0) to prepare the dataset for machine learning models:

```python
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype(int)

```

---

## 🚀 Getting Started

To run this notebook locally or on Google Colab:

1. Clone this repository or open the notebook in Google Colab.
2. Ensure you have the required libraries installed:
```bash
pip install pandas numpy matplotlib seaborn

```


3. Run the cells sequentially to reproduce the exploratory data analysis and preprocessing steps.

```

```
