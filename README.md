# Heart Disease Prediction System

An end-to-end Machine Learning web application that predicts the risk of heart disease based on patient diagnostic demographics and clinical features.

## Project Structure
- `heart.csv` : Standardized heart disease clinical dataset.
- `train_model.py` : Backend script for preprocessing, standard scaling, and Logistic Regression training.
- `app.py` : Frontend production logic rendering the interactive Streamlit user dashboard.
- `*.pkl` : Serialized pipeline model components (`heart_model.pkl`, `scaler.pkl`, `columns.pkl`).

## Performance Matrix Metrics
- **Overall Model Accuracy**: 84.78%
- **Precision**: 0.9072
- **Recall**: 0.8224
- **F1 Score**: 0.8627

## How to Execute Locally
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
   ```
2. Train backend model artifacts:
   ```bash
   python train_model.py
   ```
3. Boot up frontend Streamlit web engine UI:
   ```bash
   streamlit run app.py
   ```
