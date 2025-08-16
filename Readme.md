# EquityAI- Public and Private Equity Prediction System

## About the Project
The **EquityAI- Public and Private Equity Prediction System** is a Python-based web application built with **Streamlit** that allows users to simulate private equity investments, estimate returns, and evaluate risk. The application calculates key financial metrics such as **Internal Rate of Return (IRR)**, **expected final value**, and **failure probability** for different investment scenarios.  

This project helps investors, students, and financial analysts to understand the dynamics of investment performance under uncertainty and make data-driven decisions.

---

## Features
- Predict IRR% for investments based on input parameters.
- Calculate **expected, median, and percentile values** of the final investment.
- Evaluate **failure probability** for a given investment scenario.
- Interactive web interface using **Streamlit**.
- Visualizations and summary statistics for better analysis.

---

## Evaluation Metrics & Performance

The performance of the investment predictions is measured using standard regression metrics:

| Metric | Public Equity (Market Returns) | Private Equity (IRR Prediction) |
|--------|-------------------------------|--------------------------------|
| MAE    | 24.9556                       | 20.8360                        |
| RMSE   | 28.6632                       | 26.1753                        |
| R²     | -2.6521                       | 0.7587                         |

> The negative R² for Public Equity indicates that the model performs worse than a horizontal mean predictor, while the Private Equity IRR prediction shows strong predictive power with an R² of 0.7587.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/TejasNeswi/Investment-Return-Simulator.git
cd Investment-Return-Simulator
```
2. **Create a virtual environment:**
```bash
python -m venv venv
```
3. **Activate the virtual environment:**
```bash
venv\Scripts\activate
```
4. **Install dependencies:**
```bash
pip install -r requirements.txt
```
5. **Run the streamlit app:**
```bash
streamlit run app.py
```
6. **Open the browser at the provided local URL (usually http://localhost:8501).**

7. **Enter investment parameters and explore the predicted IRR, expected final value, and probability of failure.**

---

## Technologies Used:

- Python

- Streamlit

- pandas, numpy

- scikit-learn (for predictive models)

- matplotlib / seaborn (for visualizations)

---

## License

This project is open source and available under the MIT License.


