import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create mock data for house price prediction
np.random.seed(42)

# Features: size (sqft), number of rooms, age of house (years)
n_samples = 1000
house_size = np.random.uniform(500, 4000, size=n_samples)  # House size in sqft
num_rooms = np.random.randint(1, 10, size=n_samples)  # Number of rooms
house_age = np.random.randint(1, 100, size=n_samples)  # Age of house in years

# Target: House price (in $1000)
house_price = (house_size * 0.1) + (num_rooms * 5000) - (house_age * 100) + np.random.normal(0, 50000, size=n_samples)

# Stack features into one matrix
X = np.column_stack((house_size, num_rooms, house_age))
y = house_price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model
def predict_price(house_size_input, num_rooms_input, house_age_input):
    # Prepare input data
    input_data = np.array([[house_size_input, num_rooms_input, house_age_input]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit App
st.title('House Price Prediction App')

st.header('Enter the house details to predict the price')

# Create input fields for the user to enter data
house_size_input = st.number_input('Enter house size (in sqft)', min_value=500, max_value=5000, value=1500)
num_rooms_input = st.number_input('Enter number of rooms', min_value=1, max_value=10, value=3)
house_age_input = st.number_input('Enter age of house (in years)', min_value=1, max_value=100, value=10)

# Button to predict price
if st.button('Predict Price'):
    predicted_price = predict_price(house_size_input, num_rooms_input, house_age_input)
    st.write(f'The predicted house price is: ${predicted_price:,.2f}')
