# sales
# Retail Demand Forecasting using LightGBM with Cross-Product Promotion Elasticity (CPPE)

## Project Overview

This project predicts future product demand in retail stores using machine learning.

Traditional sales forecasting models mainly rely on historical sales, promotions, and seasonal trends. This project introduces an additional feature called **Cross-Product Promotion Elasticity (CPPE)** to capture how promotions on one product can influence the sales of another product.

Examples:

- Discount on cheap chocolate → premium chocolate sales may decrease
- Bread promotion → butter sales may increase
- Electronics discount → accessory sales may increase

This helps the model learn relationships between products instead of treating each product independently.

---

## Objective

Build an intelligent retail demand forecasting system that can:

- Predict future sales quantity
- Capture historical sales patterns
- Understand promotion effects
- Identify cross-product influence
- Reduce stock shortages and overstock situations
- Generate future forecasts

---

## Dataset

Dataset: Retail Demand Forecasting Dataset

Features used:

- Date
- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Demand Forecast
- Price
- Discount
- Weather Condition
- Holiday/Promotion
- Competitor Pricing
- Seasonality

---

## Feature Engineering

### Date Features

Extracted from Date column:

- Year
- Month
- DayOfWeek

Purpose:

Allows the model to learn:

- Seasonal patterns
- Monthly trends
- Weekly buying behavior

---

### Lag Features

Created:

- Units Sold_lag1
- Units Sold_lag2
- Units Sold_lag3

Purpose:

Adds historical memory to the model.

Example:

| Day | Units Sold |
|------|-------------|
| Monday | 100 |
| Tuesday | 120 |
| Wednesday | 90 |

For Wednesday:

- lag1 = 120
- lag2 = 100
- lag3 = 0

---

### Cross-Product Promotion Elasticity (CPPE)

Custom feature introduced in this project.

Purpose:

Measures how promotions of one product influence sales of another product.

Examples:

| Product A | Product B Promotion | Effect |
|------------|---------------------|---------|
| Premium Chocolate | Cheap Chocolate | Sales decrease |
| Butter | Bread | Sales increase |
| Mobile Accessories | Smartphones | Sales increase |

CPPE captures:

- Product relationships
- Promotion impact
- Complementary behavior
- Cannibalization effects

---

## Model Used

LightGBM Regressor (LGBM)

Reason for choosing LightGBM:

- Fast training
- Efficient for large datasets
- Supports categorical variables
- Handles complex feature interactions
- High prediction performance

Model Parameters:

```python
model = lgb.LGBMRegressor(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42
)
```

---

## Workflow

```text
Retail Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Date Features
        ↓
Lag Features
        ↓
CPPE Feature
        ↓
Time-based Train-Test Split
        ↓
LightGBM Model
        ↓
Sales Prediction
        ↓
SHAP Analysis
        ↓
30-Day Future Forecast
```

---

## Model Explainability

SHAP (SHapley Additive Explanations) was used to understand:

- Which features contribute most to predictions
- Positive and negative impacts of features
- Promotion influence
- Cross-product effects

Visualization:

- SHAP violin plot
- Feature importance analysis

---

## Results

Performance Metrics:

```text
MAE  : 7.22
RMSE : 8.46
R²   : 0.9939
```

The model successfully learns:

- Historical demand patterns
- Promotion behavior
- Product interactions
- Seasonal effects

---

## Future Forecasting

The model generates future demand predictions and creates a:

- 30-day future sales forecast
- Daily forecast visualization

This helps in:

- Inventory planning
- Stock management
- Reducing losses

---

## Technologies Used

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-learn
- SHAP
- Matplotlib

---

## Future Improvements

- Dynamic CPPE using time-decay
- Real-time forecasting
- Streamlit dashboard
- Hyperparameter optimization
- External factors integration

---

## Author

J Janasthuthi
