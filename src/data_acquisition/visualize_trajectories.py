import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load Data
FILE_PATH = os.path.join("..", "..", "data", "raw", "simulated_trajectories.csv")

try:
    df = pd.read_csv(FILE_PATH)
    print(f" Data loaded: {len(df)} rows.")
except FileNotFoundError:
    print("CSV file not found. Please run the trajectory generator first.")
    exit()

def plot_scenario(scenario_id):
    # 2. Filter Data for Given Scenario ID
    data = df[df["id"] == scenario_id]
    
    # Extract label (all rows have the same label for a scenario)
    is_conflict = data["label"].iloc[0] == 1
    label_text = " CONFLICT" if is_conflict else " SAFE"
    color_text = "red" if is_conflict else "green"

    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Aircraft 1 (Blue)
    plt.plot(data["x1"], data["y1"], label="Aircraft 1", color="blue", linewidth=2)
    plt.scatter(data["x1"].iloc[0], data["y1"].iloc[0], color="blue", marker="^", s=100, label="Start A1")
    
    # Aircraft 2 (Red)
    plt.plot(data["x2"], data["y2"], label="Aircraft 2", color="orange", linewidth=2)
    plt.scatter(data["x2"].iloc[0], data["y2"].iloc[0], color="orange", marker="^", s=100, label="Start A2")

    # Title and details
    plt.title(f"Scenario #{scenario_id} - {label_text}", color=color_text, fontsize=16, fontweight='bold')
    plt.xlabel("Position X (meters)")
    plt.ylabel("Position Y (meters)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.axis('equal') # Maintain real proportions (to avoid distorting distances)
    
    plt.show()

# --- RUN ---
# Looking for an example of conflict and one safe
conflict_id = df[df["label"] == 1]["id"].unique()[0]
safe_id = df[df["label"] == 0]["id"].unique()[0]

print(f"Displaying Conflict Scenario ID: {conflict_id}")
plot_scenario(conflict_id)

print(f"Displaying Safe Scenario ID: {safe_id}")
plot_scenario(safe_id)