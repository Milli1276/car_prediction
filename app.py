import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

# --- 1. DATA PROCESSING & ACCURACY TESTING ---
def train_model():
    # Load and Preprocess - Fixed filename to car_data.csv
    df = pd.read_csv('car_data.csv') 
    df['Age'] = 2026 - df['Year']
    df_model = df.drop(['Car_Name', 'Year'], axis=1)
    df_model = pd.get_dummies(df_model, drop_first=True)
    
    X = df_model.drop('Selling_Price', axis=1)
    y = df_model['Selling_Price']
    
    # Split for Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # --- 2. SAVE GRAPH (Object-Oriented Fix) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f'Model Accuracy (R2 Score: {r2:.2f})')
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    plt.tight_layout()
    fig.savefig('accuracy_graph.png') 
    plt.close(fig) 
    
    return model, X.columns, r2, mse

# --- 3. WEB APP INTERFACE ---
def main():
    st.set_page_config(page_title="Car Price AI")
    st.title("🚗 Car Price Analysis & Prediction")

    # Call the training function
    try:
        model, model_columns, r2, mse = train_model()

        # Display Metrics & Graph in the App
        st.header("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy (R2)", f"{r2:.2f}")
        col2.metric("Mean Squared Error", f"{mse:.2f}")
        
        st.image('accuracy_graph.png', caption="Visual Accuracy Graph")

        st.divider()

        # User Input Sidebar
        st.sidebar.header("Input Car Details")
        p_price = st.sidebar.number_input("Present Price (Lakhs)", 0.1, 100.0, 5.0)
        kms = st.sidebar.number_input("Kms Driven", 0, 500000, 20000)
        age = st.sidebar.slider("Age of Car", 0, 25, 5)
        fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
        trans = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

        # Prediction Logic
        if st.button("Predict Selling Price"):
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)
            
            input_data['Present_Price'] = p_price
            input_data['Driven_kms'] = kms
            input_data['Age'] = age
            
            if fuel == "Diesel": input_data['Fuel_Type_Diesel'] = 1
            if fuel == "Petrol": input_data['Fuel_Type_Petrol'] = 1
            if seller == "Individual": input_data['Selling_type_Individual'] = 1
            if trans == "Manual": input_data['Transmission_Manual'] = 1
            
            prediction = model.predict(input_data)[0]
            st.success(f"### Estimated Price: ₹{max(0, prediction):.2f} Lakhs")
            
    except FileNotFoundError:
        st.error("Error: 'car_data.csv' not found. Please ensure the file is in the same folder as app.py.")

if __name__ == "__main__":
    main()