import re

import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_model.pkl")

print("✅ Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("📝 Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    title = title.strip()
    broj_reci = len(title.split())
    duzina_naslova = len(title)
    ima_brojeve = int(bool(re.search(r"\d", title)))
    ima_velika_slova = int(bool(re.search(r"[A-Z]", title)))
    max_duzina_reci = max((len(r) for r in title.split()), default=0)

    user_input = pd.DataFrame(
        [
            {
                "Product Title": title.lower(),
                "broj_reci": broj_reci,
                "duzina_naslova": duzina_naslova,
                "ima_brojeve": ima_brojeve,
                "ima_velika_slova": ima_velika_slova,
                "max_duzina_reci": max_duzina_reci,
            }
        ]
    )

    prediction = model.predict(user_input)[0]
    print(f"🔎 Predicted category: {prediction}\n" + "-" * 40)
