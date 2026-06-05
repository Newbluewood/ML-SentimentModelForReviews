# ML projekat — predikcija kategorije proizvoda (kompletan pipeline)

Repozitorijum sadrži kompletan mašinsko-učenje pipeline za **predikciju kategorije proizvoda na osnovu naziva (Product Title)** koristeći Python i scikit-learn.

Projekat je razvijen u okviru edukativnog modula i pokriva sve tipične faze ML workflow-a — od sirovih podataka do treniranog modela spremnog za korišćenje.

---

## Struktura projekta

```
├── data/
│   └── IMLP4_TASK_03-products.csv       # skup podataka
├── notebooks/
│   └── product_category_analysis.ipynb  # EDA, inženjering karakteristika, treniranje i evaluacija
├── src/
│   ├── train_model.py                   # treniranje i čuvanje modela
│   ├── test_model.py                    # interaktivno testiranje (unos naziva proizvoda)
│   └── predict_category.py              # predikcija na primerima iz zadatka
├── model/
│   └── category_model.pkl               # trenirani model (generiše se lokalno)
├── requirements.txt
└── README.md
```

---

## Šta projekat sadrži

| Komponenta | Lokacija | Opis |
|------------|----------|------|
| Trenirani model (`.pkl`) | `model/category_model.pkl` | Pipeline (preprocessing + Random Forest), čuva se nakon pokretanja `train_model.py` |
| Skripta za treniranje | `src/train_model.py` | Učitava podatke, primenjuje isti tok kao notebook, trenira i čuva model |
| Interaktivno testiranje | `src/test_model.py` | Korisnik unosi naziv proizvoda, model predviđa kategoriju |
| Jupyter sveska | `notebooks/product_category_analysis.ipynb` | Kompletna analiza, FE, treniranje (LR, NB, RF) i matrica zabune |
| Dokumentacija | `README.md` | Uputstvo za instalaciju, treniranje i testiranje |

> **Napomena o modelu:** Fajl `category_model.pkl` nije u repozitorijumu (prevelik za GitHub). Nakon kloniranja pokrenite `python src/train_model.py` — model se kreira za nekoliko minuta i čuva u `model/`.

---

## Koraci u modulu

### 1. Postavljanje projekta
- Kreiran GitHub repozitorijum
- Definisana struktura foldera
- Učitan skup podataka sa ~35 000 proizvoda

### 2. Istraživanje podataka (EDA)
- Pregled oblika skupa, tipova kolona i nedostajućih vrednosti
- Vizuelizacije raspodele kategorija (`matplotlib`, `seaborn`)
- Analiza kolone `Product Title` i numeričkih kolona

### 3. Čišćenje i predobrada
- Uklanjanje redova sa nedostajućim vrednostima (`dropna`)
- Standardizacija naziva kategorija (npr. `fridge` → `Fridges`)
- Uklanjanje suvišnih kolona (ID-jevi, datumi, metapodaci prodavca)

### 4. Inženjering karakteristika
- `broj_reci`, `duzina_naslova`, `ima_brojeve`, `ima_velika_slova`, `max_duzina_reci`
- Pretvaranje naslova u mala slova
- Ulaz modela: `Product Title` + numeričke karakteristike

### 5. Treniranje i evaluacija
- Poređenje modela: Logistic Regression, Naive Bayes, Random Forest
- `TfidfVectorizer`, `MinMaxScaler`, `ColumnTransformer`, `Pipeline`
- Evaluacija: tačnost i matrica zabune
- **Random Forest** — najbolji rezultat na test skupu

### 6. Finalni model
- Treniranje na celom očišćenom skupu
- Čuvanje pipeline-a pomoću `joblib` u `model/category_model.pkl`

### 7. Inferencija
- Učitavanje sačuvanog modela
- Interaktivno testiranje preko konzole
- Skripta sa primerima naslova iz zadatka

---

## Kako koristiti

Sve komande pokretati iz **korena projekta** (`Task3/`).

### 1. Kloniranje repozitorijuma

```bash
git clone git@github.com:Newbluewood/ML-SentimentModelForReviews.git
cd ML-SentimentModelForReviews
```

### 2. Instalacija zavisnosti

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Treniranje modela

```bash
python src/train_model.py
```

Kreira fajl `model/category_model.pkl`.

### 4. Interaktivno testiranje

```bash
python src/test_model.py
```

Unesite naziv proizvoda kada se zatraži. Za izlaz ukucajte `exit`.

Primer:

```
Unesite naziv proizvoda: iphone 7 32gb gold
Predviđena kategorija: Mobile Phones
```

### 5. Predikcija na primerima iz zadatka

```bash
python src/predict_category.py
```

Ili sa proizvoljnim naslovom:

```bash
python src/predict_category.py "bosch wap28390gb 8kg 1400 spin"
```

### 6. Jupyter sveska

Otvorite `notebooks/product_category_analysis.ipynb` u Jupyter-u ili Google Colab-u za kompletnu analizu korak po korak.

---

## Autor

Projekat je razvijen u okviru programa **Introduction to Machine Learning using Python** (IT Academy).

**Autor:** Nebojša Simović

## Licenca

Projekat je otvorenog koda i slobodan za edukativnu upotrebu.
