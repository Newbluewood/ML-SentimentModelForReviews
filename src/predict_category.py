import re
import sys

import joblib
import pandas as pd

model = joblib.load("model/category_model.pkl")

# Primeri naslova za proveru modela
TEST_TITLES = [
    "iphone 7 32gb gold,4,3,Apple iPhone 7 32GB",
    "olympus e m10 mark iii geh use silber",
    "kenwood k20mss15 solo",
    "bosch wap28390gb 8kg 1400 spin",
    "bosch serie 4 kgv39vl31g",
    "smeg sbs8004po",
]


def predict_category(title):
    title = title.strip()
    user_input = pd.DataFrame(
        [
            {
                "Product Title": title.lower(),
                "broj_reci": len(title.split()),
                "duzina_naslova": len(title),
                "ima_brojeve": int(bool(re.search(r"\d", title))),
                "ima_velika_slova": int(bool(re.search(r"[A-Z]", title))),
                "max_duzina_reci": max((len(r) for r in title.split()), default=0),
            }
        ]
    )
    return model.predict(user_input)[0]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
        print(f"🔎 Predicted category: {predict_category(title)}")
    else:
        print("✅ Model loaded. Predictions for test titles:\n")
        for title in TEST_TITLES:
            prediction = predict_category(title)
            print(f"Title:    {title}")
            print(f"Category: {prediction}\n" + "-" * 40)
