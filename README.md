# Autism Prediction 🧠🧬

A machine learning–based application that predicts the likelihood of Autism Spectrum Disorder (ASD) in adults based on answers to a screening questionnaire and demographic inputs. This project streamlines data preprocessing, model training, evaluation, and live predictions using saved model pipelines.

---

## 🌟 Project Highlights

- **Goal**: Predict autism tendency using behavioral responses and metadata (age, gender, etc.).
- **Dataset**: Public dataset with anonymized adult screening data.
- **Model**: Supervised classification with serialized `.pkl` pipeline.
- **Use Case**: Early-stage autism screening and decision-support tool.

---

## 🧠 Core Components

### 1. Dataset Overview (`train.csv`)

| Feature             | Description                                      |
|---------------------|--------------------------------------------------|
| `A1_Score` to `A10_Score` | Responses to 10 binary autism screening questions |
| `age`               | Age of participant                               |
| `gender`            | Gender (M/F)                                     |
| `ethnicity`         | Ethnic background                                |
| `jaundice`          | Born with jaundice? (`yes`/`no`)                 |
| `family_mem_with_ASD` | Family history of ASD (`yes`/`no`)            |
| `used_app_before`   | Previously used autism screening app             |
| `country_of_res`    | Country of residence                             |
| `result`            | Screening score                                  |
| `age_desc`          | Age group description (e.g. ‘Adult’)             |
| `relation`          | Relation to the respondent (e.g. ‘Parent’)       |
| `Class/ASD`         | Target label (Yes/No – autism tendency)          |

---

### 2. Model Pipeline

- **Preprocessing**:
  - Categorical Encoding via `LabelEncoder` or `OneHotEncoder`
  - Normalization of numeric features
  - Handling of missing/null values
- **Model**:
  - Random Forest Classifier (best performing)
  - Exported via `joblib` or `pickle` to `best_model.pkl`
- **Encoders**:
  - Stored separately in `encoders.pkl` for decoding input features

---

## 📊 Sample Results

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | ~91.2%    |
| Precision     | ~89.5%    |
| Recall        | ~92.0%    |
| F1 Score      | ~90.7%    |

> Example Output:
```python
Input: {'gender': 'M', 'age': 24, 'A1_Score': 1, ..., 'used_app_before': 'yes'}
Prediction: Yes (Likely Autism)
```

---

## 📁 Project Structure

```
autism-prediction/
│
├── data/
│   └── train.csv                     # Dataset used for training
│
├── notebook/
│   ├── autism_preidiction.ipynb     # Full notebook for preprocessing + prediction
│   ├── best_model.pkl               # Trained and saved classifier
│   └── encoders.pkl                 # Label encoders used during training
│
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** – Modeling and pipeline creation
- **pandas / NumPy** – Data wrangling
- **joblib / pickle** – Model serialization
- **Matplotlib / Seaborn** – (Optional) data visualization
- **Jupyter Notebook**

---

## 🚀 Future Enhancements

- Add an **interactive web form** using Streamlit or Gradio
- Build a **REST API** with Flask/FastAPI for integration
- Implement **SHAP** or **LIME** for explainable predictions
- Extend dataset to other age groups (children/teens)
- Enable multilingual input support for international use

---

## 📌 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit.io](https://streamlit.io/)
- [LIME for Model Explainability](https://github.com/marcotcr/lime)
