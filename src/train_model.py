from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/IMLP4_TASK_03-products.csv")
df.columns = df.columns.str.strip()

# Drop all rows with missing values
df = df.dropna()

# Standardize category labels
df["Category Label"] = df["Category Label"].astype(str).str.strip()
df["Category Label"] = df["Category Label"].replace({
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
})
df["Category Label"] = df["Category Label"].astype("category")

# Clean title and create numeric features from title
df["Product Title"] = df["Product Title"].astype(str).str.strip()
df["broj_reci"] = df["Product Title"].str.split().str.len()
df["duzina_naslova"] = df["Product Title"].str.len()
df["ima_brojeve"] = df["Product Title"].str.contains(r"\d", regex=True).astype(int)
df["ima_velika_slova"] = df["Product Title"].str.contains(r"[A-Z]", regex=True).astype(int)
df["max_duzina_reci"] = df["Product Title"].str.split().apply(
    lambda reci: max((len(r) for r in reci), default=0)
)
df["Product Title"] = df["Product Title"].str.lower()

# Drop columns that are not useful for modeling
df = df.drop(columns=[
    "product ID",
    "Merchant ID",
    "_Product Code",
    "Number_of_Views",
    "Merchant Rating",
    "Listing Date",
])

# Features and label
X = df[[
    "Product Title",
    "broj_reci",
    "duzina_naslova",
    "ima_brojeve",
    "ima_velika_slova",
    "max_duzina_reci",
]]
y = df["Category Label"]

# Preprocessor: TF-IDF for title, MinMaxScaler for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title"),
        ("numeric", MinMaxScaler(), [
            "broj_reci",
            "duzina_naslova",
            "ima_brojeve",
            "ima_velika_slova",
            "max_duzina_reci",
        ]),
    ]
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
Path("model").mkdir(exist_ok=True)
joblib.dump(pipeline, "model/category_model.pkl")

print("✅ Model trained and saved as 'model/category_model.pkl'")
