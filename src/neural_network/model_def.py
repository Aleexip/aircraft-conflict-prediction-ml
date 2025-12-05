import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_conflict_model(input_shape=(30, 14)):
 
    model = Sequential([
        # Input Layer
        Input(shape=input_shape),
        
        # Layer 1 LSTM: returns sequences to feed into next LSTM
        # It is the memmory layer that captures temporal dependencies
        LSTM(64, return_sequences=True),
        Dropout(0.2), # Prevent overfitting
        
        # Layer 2 LSTM: processes sequences from previous layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense Layer: Final processing
        Dense(16, activation='relu'),
        
        # Output Layer: Single neuron with Sigmoid
        # 0 = Safe, 1 = Conflict
        Dense(1, activation='sigmoid')
    ])

    # Compiling the model (choosing the "teacher" and the "grading method")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# This part runs only if this script is executed directly
if __name__ == "__main__":
    model = build_conflict_model()
    model.summary()
    print(" The model has been defined and compiled successfully!")