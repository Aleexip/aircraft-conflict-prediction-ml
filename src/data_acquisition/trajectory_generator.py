import numpy as np
import pandas as pd
import os
from datetime import datetime

# Configuration
NUM_SCENARIOS = 2000     # Number of flight pairs to generate
DURATION = 120           # Duration of each scenario (seconds)
DT = 1                   # Time step (1 second)
CONFLICT_RATIO = 0.5     # 50% conflicts, 50% safe
MIN_DIST_CONFLICT = 9260 # 5 NM (Nautical Miles) in meters - collision threshold
#Where i am saving the data
# I ensure that I exit 'src/data_acquisition' and enter 'data/raw'
OUTPUT_PATH = os.path.join("..", "..", "data", "raw", "simulated_trajectories.csv")

def generate_encounter(scenario_id, force_conflict=False):
    """
    Generates a scenario with 2 aircraft.
    If force_conflict=True, we force the trajectories to intersect.
    """
    # 1. Initialize Aircraft 1 (relatively straight)
    x1, y1 = np.random.uniform(-50000, 50000), np.random.uniform(-50000, 50000)
    speed1 = np.random.uniform(200, 250) # m/s
    heading1 = np.random.uniform(0, 2 * np.pi)
    
    # 2. Initialize Aircraft 2
    # If we want conflict, we put it on a collision trajectory
    if force_conflict:
        # Calculate a future point of Aircraft 1
        t_impact = np.random.uniform(30, DURATION - 30)
        future_x1 = x1 + speed1 * np.cos(heading1) * t_impact
        future_y1 = y1 + speed1 * np.sin(heading1) * t_impact
        
        speed2 = np.random.uniform(200, 250)
        # Choose a start position for Aircraft 2 so that it reaches future_x1 at t_impact
        # Add some noise so it's not a perfect mathematical collision (realism)
        noise_x = np.random.normal(0, 100) 
        noise_y = np.random.normal(0, 100)
        
        # Back-calculate start position for Aircraft 2
        heading2 = np.random.uniform(0, 2 * np.pi) # Initially random, but we adjust it
        # Simplified approach: point directly towards impact point
        angle_to_impact = np.random.uniform(0, 2*np.pi)
        dist_start = speed2 * t_impact
        
        x2 = future_x1 - speed2 * np.cos(angle_to_impact) * t_impact + noise_x
        y2 = future_y1 - speed2 * np.sin(angle_to_impact) * t_impact + noise_y
        heading2 = angle_to_impact # Points towards impact
        
    else:
        # Safe Scenario (random)
        x2, y2 = np.random.uniform(-50000, 50000), np.random.uniform(-50000, 50000)
        speed2 = np.random.uniform(200, 250)
        heading2 = np.random.uniform(0, 2 * np.pi)

    # 3. Step-by-step simulation
    data = []
    
    # Turn rate factors - some will turn slightly
    turn_rate1 = np.random.normal(0, 0.005) # rad/sec
    turn_rate2 = np.random.normal(0, 0.005)
    
    curr_x1, curr_y1, curr_h1 = x1, y1, heading1
    curr_x2, curr_y2, curr_h2 = x2, y2, heading2
    
    min_dist = float('inf')
    
    for t in range(DURATION):
        # Update positions
        curr_x1 += speed1 * np.cos(curr_h1) * DT
        curr_y1 += speed1 * np.sin(curr_h1) * DT
        curr_h1 += turn_rate1 * DT # Apply turn rate
        
        curr_x2 += speed2 * np.cos(curr_h2) * DT
        curr_y2 += speed2 * np.sin(curr_h2) * DT
        curr_h2 += turn_rate2 * DT
        
        # Calculate Euclidean distance
        dist = np.sqrt((curr_x1 - curr_x2)**2 + (curr_y1 - curr_y2)**2)
        if dist < min_dist:
            min_dist = dist
            
        # Add to data list
        # Format: [ID, Time, x1, y1, speed1, heading1, x2, y2, speed2, heading2, Label]
        # Temporarily set label to 0, will set final label based on min_dist
        data.append([scenario_id, t, curr_x1, curr_y1, speed1, curr_h1, 
                     curr_x2, curr_y2, speed2, curr_h2])

    # 4. Determine Final Label
    label = 1 if min_dist < MIN_DIST_CONFLICT else 0
    
    # Add label to each row in this scenario
    final_rows = [row + [label] for row in data]
    return final_rows

def main():
    print(f" Starting generation of {NUM_SCENARIOS} scenarios...")
    all_data = []
    
    for i in range(NUM_SCENARIOS):
        # Alternate between conflict and safe flights
        force_conflict = (i < NUM_SCENARIOS * CONFLICT_RATIO)
        scenario_data = generate_encounter(i, force_conflict)
        all_data.extend(scenario_data)
        
        if i % 100 == 0:
            print(f"Generated {i} / {NUM_SCENARIOS}")

        # Create DataFrame
    columns = ["id", "time", "x1", "y1", "v1", "h1", "x2", "y2", "v2", "h2", "label"]
    df = pd.DataFrame(all_data, columns=columns)
    
    # Check output folder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Done! Data saved to: {os.path.abspath(OUTPUT_PATH)}")
    print(f"Class distribution: \n{df['label'].value_counts()}")

if __name__ == "__main__":
    main()