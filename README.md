# TensorFlow Time Series Forecasting Using Neural Networks

## 📌 Project Overview

This project demonstrates **time series forecasting using TensorFlow and a Feedforward Neural Network**. It generates synthetic time-series data containing a trend, seasonal patterns, and random noise, then trains a neural network to predict future values based on previous observations.

The project uses Python, TensorFlow, NumPy, Pandas, Matplotlib, and Scikit-learn to preprocess data, train the model, evaluate forecasting accuracy, and visualize the results.

## 🎯 Objectives

- Generate synthetic time-series data with trend, seasonality, and noise.
- Visualize time-series data using Matplotlib.
- Split the dataset chronologically into training and testing sets.
- Normalize data using MinMaxScaler.
- Create input-output sequences using a sliding window.
- Build and train a neural network using TensorFlow and Keras.
- Evaluate prediction performance using Mean Absolute Error (MAE).
- Visualize actual versus predicted values.
- Analyze training and validation loss.

## 🛠️ Technologies Used

- **Python** — Programming language
- **NumPy** — Numerical computations and data generation
- **Pandas** — Data handling
- **Matplotlib** — Data visualization
- **TensorFlow / Keras** — Neural network development and training
- **Scikit-learn** — Data normalization and model evaluation

## 📂 Project Structure

```text
TensorFlow-Time-Series-Forecasting/
│
├── time_series_forecasting.py
├── README.md
└── requirements.txt
```

*Note: Replace `time_series_forecasting.py` with the actual name of your Python script if it differs.*

## 📊 Dataset Description

The project generates a synthetic time series consisting of 300 observations.

The data is created using the following mathematical expression:

\[
Value_t = 10 + 0.03t + 2\sin\left(\frac{2\pi t}{20}\right) + \epsilon_t
\]

Where:

- **10** represents the baseline value.
- **0.03t** introduces a gradual upward trend.
- **Seasonal component** creates repeating patterns with a period of 20 time steps.
- **Random noise** introduces small variations in the observations.

The noise is sampled from a normal distribution with a mean of 0 and a standard deviation of 0.3.

A random seed of 42 is used to make the synthetic data generation reproducible.

## ⚙️ Methodology

### 1. Data Generation

NumPy generates a synthetic time series with trend, seasonality, and random noise. Pandas stores the generated observations in a Series.

### 2. Data Visualization

Matplotlib plots the original time series to help identify its trend, seasonal behavior, and fluctuations.

### 3. Train-Test Split

The dataset is divided chronologically:

- **Training set:** First 80% of observations (240 points).
- **Testing set:** Final 20% of observations (60 points).

This preserves the temporal order of the data.

### 4. Data Normalization

MinMaxScaler transforms the training observations to a range between 0 and 1. The same fitted scaler is applied to the testing observations.

Fitting the scaler only on training data helps prevent information leakage from the test set.

### 5. Sequence Preparation

A sliding window of 10 time steps is used to construct the model's inputs.

Each input contains the previous 10 observations, while the target is the next observation.

For example:

```text
Input:  [Value1, Value2, ..., Value10]
Target: Value11
```

### 6. Neural Network Architecture

The project uses a feedforward neural network built with Keras Sequential.

| Layer | Configuration |
|---|---|
| Input | 10 time steps |
| Hidden Layer 1 | 64 neurons, ReLU activation |
| Hidden Layer 2 | 32 neurons, ReLU activation |
| Output Layer | 1 neuron |

The network learns patterns from the input sequences and produces a prediction for the next time step.

### 7. Model Compilation and Training

The model is compiled with:

- **Optimizer:** Adam
- **Loss function:** Mean Squared Error (MSE)
- **Epochs:** 50
- **Batch size:** 16
- **Validation split:** 10% of the training sequences

The model learns by minimizing the difference between predicted and actual normalized values.

### 8. Prediction and Evaluation

The trained model generates predictions for the test sequences. Both predictions and actual target values are converted back to the original scale.

The model is evaluated using Mean Absolute Error (MAE).

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

A lower MAE indicates that predictions are closer to the actual observations on average.

### 9. Visualization of Results

The project generates three plots:

1. **Original Time Series:** Displays the generated data.
2. **Actual vs. Predicted Values:** Compares the model's forecasts against actual test observations.
3. **Training vs. Validation Loss:** Shows how model errors change during training.

## 📦 Installation

### Prerequisites

- Python 3.10–3.12 is a reasonable starting point; check the TensorFlow installation guide for platform-specific compatibility.
- pip package manager
- Visual Studio Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
tensorflow
scikit-learn
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

Execute the Python script from your terminal:

```bash
python time_series_forecasting.py
```

The program will:

1. Generate the synthetic dataset.
2. Display the original time-series plot.
3. Train the neural network.
4. Print the test MAE.
5. Display the actual-versus-predicted plot.
6. Display the training and validation loss plot.

## 📈 Expected Results

After execution, you should observe:

- A time series with an upward trend and repeating seasonal patterns.
- Training progress showing loss values across epochs.
- A printed test MAE value.
- A plot comparing actual observations with predicted values.
- A plot showing training and validation loss.

**Note:** The exact MAE and plot appearance depend on the software environment and training behavior. Run the program to obtain the actual results; no specific accuracy is guaranteed.

## 💡 Applications

Time-series forecasting techniques can be applied to:

- Sales and demand forecasting
- Weather and temperature prediction
- Stock and financial time-series analysis
- Energy consumption forecasting
- Traffic volume prediction
- Sensor data analysis
- Business trend analysis

This project uses synthetic data for demonstration and is not a validated forecasting system for real-world applications.

## 🔍 Limitations

- The dataset is synthetic rather than collected from a real-world source.
- The model uses a simple feedforward neural network rather than a specialized recurrent or temporal architecture.
- Only 300 observations are generated, which limits the complexity of patterns the model can learn.
- The model predicts one time step at a time.
- Performance may vary depending on the data, model architecture, and training settings.
- The test sequence preparation does not use the final training observations as context for the first test prediction, which can reduce the usefulness of the evaluation.

## 🚀 Future Improvements

- Use real-world time-series datasets.
- Compare the feedforward neural network with LSTM, GRU, and other forecasting models.
- Experiment with different look-back windows.
- Add early stopping and model checkpointing.
- Evaluate additional metrics such as RMSE and MAPE.
- Improve test sequence preparation by using the final training observations as context.
- Tune hyperparameters to improve forecasting performance.
- Save and reload the trained model for future predictions.


---

⭐ If you find this project useful, consider giving the repository a star!
