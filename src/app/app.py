import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# We add the project root to sys.path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.neural_network.model_def import build_conflict_model

# Configuration Streamlit page
st.set_page_config(page_title="SIA - Conflict Detection", layout="wide")

st.title(" SIA: Detection of Aircraft Conflicts")
st.markdown("**System based on Neural Networks (LSTM)**")

# --- LOAD DATA ---
DATA_PATH = os.path.join("data", "raw", "simulated_trajectories.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Control Panel")

if df is not None:
    st.sidebar.success("✅ Data loaded successfully!")
    
    # Select Scenario
    all_ids = df['id'].unique()
    selected_id = st.sidebar.selectbox("Select a Flight Scenario (ID):", all_ids)
    
    # Filter data for the selected scenario
    scenario_data = df[df['id'] == selected_id]
    
    # Display Scenario Info
    is_conflict_real = scenario_data['label'].iloc[0] == 1
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Real Status:**")
    if is_conflict_real:
        st.sidebar.error("⚠️ CONFLICT")
    else:
        st.sidebar.success("✅ SAFE")

else:
    st.sidebar.error("❌ CSV file is missing! Run the generator first.")
    st.stop()

# --- MAIN AREA ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Trajectory Visualization (ID: {selected_id})")
    
    # Draw the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Aircraft 1
    ax.plot(scenario_data['x1'], scenario_data['y1'], label='Aircraft 1', color='blue')
    ax.scatter(scenario_data['x1'].iloc[0], scenario_data['y1'].iloc[0], marker='^', color='blue', s=100, label='Start A1')
    
    # Aircraft 2
    ax.plot(scenario_data['x2'], scenario_data['y2'], label='Aircraft 2', color='orange')
    ax.scatter(scenario_data['x2'].iloc[0], scenario_data['y2'].iloc[0], marker='^', color='orange', s=100, label='Start A2')
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis('equal')
    
    st.pyplot(fig)

with col2:
    st.subheader("AI Analysis")
    st.write("Load the model and run prediction on this scenario.")
    
    if st.button("🚀 Run Prediction"):
        # 1. Load architecture      
        model = build_conflict_model()
        
        # 2. Simulate input (Currently random data, until full preprocessing is done)
        # Model expects (1, 30, 14)
        dummy_input = np.random.rand(1, 30, 14)
        
        # 3. Prediction
        prediction = model.predict(dummy_input)[0][0]
        
        st.write("---")
        st.write(f"**Collision Probability:** {prediction:.2%}")
        
        if prediction > 0.5:
            st.error("ALERT: High Risk!")
        else:
            st.success("Prediction: Safe Flight")
            
        st.info("Note: The model is untrained (random weights), so the prediction is currently random.")

# --- FOOTER ---
st.markdown("---")
st.caption("Neural Networks Project - Stage 4 - Functional Architecture")