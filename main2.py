import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit app design
st.set_page_config(page_title="7-Day Temperature Forecast", page_icon="🌤️", layout="centered")
st.title("🌤️ 7-Day Temperature Forecast In Woxsen")
st.write("Get the upcoming week's temperature forecast powered by ARIMA modeling.")
st.markdown("""
    <style>
    .main {
        background-color: #e6f7ff;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput, .stTextInput input, .stButton button {
        border-radius: 5px;
        background-color: #f0f5ff;
        color: #005b96;
    }
    body, .stApp {
        color: #000; /* Set the font color to black */
    }
    </style>
""", unsafe_allow_html=True)

# Specify the path to your local dataset
file_path = 'cleaned_dataset.csv'  # Replace with the actual path to your CSV file on your laptop

# Load dataset from local file
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
st.write("### Dataset Preview")
st.dataframe(df.head())

# Choose column for prediction
column = 'InsideTemp_C'  # Replace with your actual column name
temperature_data = df[column].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(temperature_data)

# Function to create dataset for time-series forecasting
def create_dataset(df, look_back=1):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[i:i+look_back, 0])
        y.append(df[i + look_back, 0])
    return np.array(X), np.array(y)

# Hyperparameters
look_back = 100  # Number of hours to look back (this can be adjusted)
train_ratio = 0.8  # 80% for training and 20% for testing
epochs = 1  # Number of epochs
batch_size = 32  # Batch size

# Create dataset
X, y = create_dataset(scaled_data, look_back)
train_size = int(len(X) * train_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display RMSE and MAPE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test_actual, label='True Temperature', color="blue", alpha=0.7, linewidth=2)
ax.plot(predictions, label='Predicted Temperature', color="red", alpha=0.7, linestyle="--")
ax.legend()
ax.set_title("LSTM Prediction Results", fontsize=16)
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
st.pyplot(fig)

# Predict future values
st.write("### Predict Future Values")
future_steps = st.number_input("Enter the number of future steps to predict:", min_value=1, max_value=100, value=10)
last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)  # Reshape to match LSTM input

future_predictions = []
for _ in range(future_steps):
    next_pred = model.predict(last_sequence)[0]
    future_predictions.append(next_pred)
    # Update last_sequence for next prediction
    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

future_predictions = scaler.inverse_transform(future_predictions)
future_predictions_df = pd.DataFrame({
    "Step": [f"T+{i+1}" for i in range(future_steps)],
    "Predicted Temperature (°C)": future_predictions.flatten()
})

st.write("### Future Predictions")
st.table(future_predictions_df)
