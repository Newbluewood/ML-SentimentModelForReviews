# Product Category Classification ML Project (Complete Pipeline)

This repository contains a complete machine learning pipeline for **predicting product categories from product titles** using Python and scikit-learn.

The project was developed as part of a learning module, where we demonstrated all typical phases of a machine learning workflow — from raw data to a ready-to-use trained model.

---

## Project Structure

```
├── data/
│   └── IMLP4_TASK_03-products.csv    # dataset
├── notebooks/
│   └── product_category_analysis.ipynb   # EDA, preprocessing, and model comparison
├── src/
│   ├── train_model.py                  # train and save the model
│   ├── test_model.py                   # interactive console testing
│   └── predict_category.py             # batch predictions on sample titles
├── model/                              # generated locally (not in repo)
│   └── category_model.pkl
└── README.md
```

---

## What We Did in This Module

Throughout this module, we covered all major steps of a real-world ML project:

### 1. Project Setup
- Created a new GitHub repository
- Defined project folder structure
- Uploaded the raw product dataset

### 2. Data Exploration
- Loaded and analyzed a dataset with ~35,000 product listings
- Used `matplotlib` and `seaborn` for visualizations
- Investigated category distribution, missing values, and title characteristics

### 3. Data Cleaning & Preprocessing
- Removed rows with missing values
- Standardized category labels (e.g. `fridge` → `Fridges`, `CPU` → `CPUs`)
- Stripped whitespace from product titles
- Dropped irrelevant columns (IDs, dates, merchant metadata)

### 4. Feature Engineering
- Created numeric features from titles: word count, title length, digits, uppercase letters, max word length
- Converted titles to lowercase for text modeling
- Selected input features: `Product Title` + engineered numeric columns

### 5. Model Training & Evaluation
- Compared multiple ML models (Logistic Regression, Naive Bayes, Random Forest)
- Used `TfidfVectorizer`, `MinMaxScaler`, `ColumnTransformer`, and `Pipeline`
- Evaluated using accuracy score and confusion matrix
- **Random Forest** achieved the best performance on the test set

### 6. Final Model Training
- Trained the final model on the full cleaned dataset
- Saved the pipeline using `joblib` to `model/category_model.pkl`

### 7. Inference & Usage
- Loaded the saved model for prediction on new product titles
- Built an interactive console interface for real-time testing
- Added a script with predefined sample titles from the assignment

---

## How to Use

Run all commands from the project root directory.

### Install dependencies

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

### Train the model

```bash
python src/train_model.py
```

This creates `model/category_model.pkl` locally.

### Run interactive inference

```bash
python src/test_model.py
```

Enter a product title when prompted. Type `exit` to quit.

### Run sample predictions

```bash
python src/predict_category.py
```

Or pass a custom title:

```bash
python src/predict_category.py "iphone 7 32gb gold"
```

### Explore the notebook

Open `notebooks/product_category_analysis.ipynb` in Jupyter or Google Colab for the full step-by-step analysis.

---

## Author

This repository was developed as part of an educational program on practical machine learning using Python.  
All steps were documented and modularized to help students understand and reproduce the entire workflow.

**Author:** Nebojša Simović

## License

This project is open-source and freely available for educational use.
