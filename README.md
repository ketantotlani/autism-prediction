# Autism Prediction 🧠🧬

A machine learning–based application that predicts the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic inputs. Built with a streamlined pipeline that includes preprocessing, encoding, model training, and real-time prediction using a saved model and encoders.

---

## 🌟 Project Highlights

- **Goal**: Predict autism likelihood using survey and behavioral inputs.
- **Model**: Supervised ML classifier trained on autism datasets.
- **Pipeline**: Encodes input features, applies trained model, outputs binary classification.
- **Deployment Ready**: Serialized with `.pkl` files for model and encoders.
- **Application**: Can be extended to healthcare assistants, early detection tools, and screening systems.

---

## 🧠 Core Components

### 1. Data Pipeline

- **Input Data**: Survey responses + metadata (age, gender, ethnicity, etc.)
- **Preprocessing**:
  - Handling missing values
  - Label encoding and one-hot encoding
- **Feature Scaling**: StandardScaler for numeric features
- **Splitting**: Train-test split for evaluation

### 2. Model

- **Model Type**: Random Forest / Logistic Regression (based on final `.pkl`)
- **Training Metrics**: Accuracy, precision, recall
- **Cross-validation**: Applied to tune parameters and validate generalization
- **Output**: Binary class (`Yes` for likely ASD, `No` for unlikely)

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
Input: [M, 22, yes, yes, no, yes, no, no, yes, Asia]
Prediction: Yes (Likely Autism)
```

---

## 📁 Project Structure

```
autism-prediction/
│
├── data/                               # Raw and preprocessed input data (optional)
│
├── notebook/
│   ├── autism_preidiction.ipynb        # Jupyter notebook for training and inference
│   ├── best_model.pkl                  # Trained ML model
│   └── encoders.pkl                    # Label encoders for categorical features
│
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** – Model building and evaluation
- **pandas / NumPy** – Data manipulation
- **joblib / pickle** – Model and encoder serialization
- **Matplotlib / Seaborn** – Visualization (optional)

---

## 🚀 Future Enhancements

- Deploy as **Flask/FastAPI** REST endpoint for real-time API inference
- Build a **Streamlit or Gradio** front-end for survey input + prediction
- Expand dataset with more diverse demographics
- Integrate explainability tools like **SHAP or LIME**
- Add **time-series behavioral tracking** for longitudinal predictions

---

## 📌 References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pickle & Joblib Serialization](https://docs.python.org/3/library/pickle.html)
- [Streamlit for ML Apps](https://docs.streamlit.io/)
