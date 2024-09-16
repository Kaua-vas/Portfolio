# Inventory Demand Forecast

This project focuses on **predicting inventory demand** using historical sales data from the **Rossmann Store Sales** dataset. The goal is to build a machine learning model to predict future sales, considering factors like promotions, store type, and holidays.

## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Next Steps](#next-steps)

## Objective

The primary objective of this project is to:
1. **Load and clean the historical sales data** from Rossmann Stores.
2. **Build and train a machine learning model** to predict future sales using features like promotions, holidays, and store types.
3. **Evaluate the performance** of the model using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
4. **Visualize the predictions** and compare them with the historical data.

## Dataset

The dataset used in this project comes from the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition on Kaggle. It consists of:
- **train.csv**: Historical sales data for each store.
- **test.csv**: Data used for predictions (sales column is not available).
- **store.csv**: Additional information about each store (e.g., type, assortment, and competition).

## Technologies Used

- **Python**: Programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For preprocessing and scaling data.
- **TensorFlow**: To build and train the neural network model.
- **Matplotlib & Seaborn**: For visualizing data and model performance.
- **Joblib**: For saving the scaler and model for later use.

## Project Structure

Here’s an explanation of each file and its purpose:

### 1. `train_model.py`
This script:
- Loads and preprocesses the training data (`train.csv` and `store.csv`).
- Merges data from stores and sales.
- Applies feature engineering (creating features like year, month, and day).
- Normalizes the data and trains a neural network to predict sales.
- Saves the trained model and the scaler.

### 2. `predict_model.py`
This script:
- Loads the test data (`test.csv`) and preprocesses it (fills missing values, creates dummy variables).
- Loads the trained model and the scaler.
- Applies the scaler to the test data and makes predictions.
- Saves the predictions in a CSV file (`submission_corrected.csv`).

### 3. `generate_clean_test.py`
This script:
- Prepares the test data by filling missing values in the "Open" column (if the store was open or closed).
- Saves the cleaned test data for further use.

### 4. `scaler_previsao_vendas.save`
This file:
- Stores the scaler used to normalize the data during training. It ensures that the same scaling is applied to the test data for accurate predictions.

### 5. `modelo_previsao_vendas_best.keras` and `modelo_previsao_vendas_final.keras`
These files:
- Store the best version of the trained neural network model (`best`) and the final version of the model after training (`final`).

### 6. `submission_updated.csv` and `submission_corrected.csv`
These files:
- Contain the predictions made by the model. `submission_corrected.csv` is the final version with NaN values replaced by 0.

## How to Run the Project

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kaua-vas/Portfolio.git
   cd Portfolio/Inventory Demand Forecast
   ```

2. **Install the required dependencies**:
   Ensure you have Python and `pip` installed. Then, install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script**:
   This will load the training data, train the model, and save it.
   ```bash
   python train_model.py
   ```

4. **Generate cleaned test data** (optional, if needed):
   This will clean the test data for further use.
   ```bash
   python generate_clean_test.py
   ```

5. **Run the prediction script**:
   This will load the test data, apply the scaler, and generate predictions.
   ```bash
   python predict_model.py
   ```

## Results

The model was trained to predict sales for the Rossmann stores based on historical data. Key metrics on the validation set include:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual sales.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual sales.

Here are some key visualizations from the project:

### Loss During Training and Validation:
![Loss Graph](./images/loss_graph.png)

### Sales Predictions vs. Historical Sales (by store):
![Sales vs Historical Graph](./images/sales_vs_historical.png)

### Sales Predictions vs. Historical Sales (by week):
![Sales vs Week Graph](./images/sales_vs_week.png)

## Next Steps

Some potential improvements that can be made to the model include:
1. **Hyperparameter tuning**: Testing different neural network architectures, learning rates, and optimizers.
2. **Feature engineering**: Adding more features such as external data (e.g., weather) to improve model accuracy.
3. **Cross-validation**: Using cross-validation techniques to ensure the model generalizes well to unseen data.
