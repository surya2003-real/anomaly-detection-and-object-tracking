import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dat = pd.read_csv('datapoints2.csv')
dat.drop(columns=['Frame'], inplace=True)  # Remove the 'Frame' column
# Define parameters
sequence_length = 20  # Adjust as needed
prediction_time = 1  # Adjust as needed
n_features = 38  # Number of features to predict

# Function to create sequences for training and testing
def create_sequences(data, sequence_length, prediction_time):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - prediction_time + 1):
        sequence = data[i:i+sequence_length]
        target = data[i+sequence_length:i+sequence_length+prediction_time]
        sequences.append(sequence)
        targets.append(target)
    
    # Discard any remaining data that cannot be reshaped
    num_sequences = len(sequences)
    num_to_discard = num_sequences % sequence_length
    if num_to_discard > 0:
        sequences = sequences[:-num_to_discard]
        targets = targets[:-num_to_discard]
    
    return np.array(sequences), np.array(targets)


# Initialize the LSTM autoencoder model
model=load_model('model.h5')

# Loop through the IDs
unique_ids = dat['ID'].unique()

for ID in unique_ids:
    data = dat[dat['ID'] == ID].copy()
    data.drop(columns=['ID'], inplace=True)  # Remove the 'ID' column
    if(len(data) < 100):
        continue
    # Scale the data to [0, 1] range
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)
    test_data_size = len(data)//10 # Adjust as needed
    # Create sequences for training and testing
    x_train, y_train = create_sequences(data_scaled, sequence_length, prediction_time)

    # Split data into train and test sets
    train_size = len(x_train) - test_data_size
    x_train, x_test = x_train[:train_size], x_train[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # Predict the next time step for the test data
    x_pred = model.predict(x_test)
    #convert x_pred to same type as y_test
    x_pred = x_pred.astype(y_test.dtype)

    # Reshape the data for inverse_transform
    x_test_original = scaler.inverse_transform(x_test.reshape(-1, n_features))
    x_pred_original = scaler.inverse_transform(x_pred.reshape(-1, n_features))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, n_features))
    print(x_test_original.shape, x_pred_original.shape, y_test_original.shape)
    # Calculate and visualize the predictions
    mse = []
    for i in range(len(y_test_original)):
        mse_seq = np.mean(np.power(y_test_original[i] - x_pred_original[i], 2))
        mse.append(mse_seq)

    mse = np.array(mse)


    threshold = mse.mean() + 3 * mse.std()
    anomalies = (mse > threshold).astype(int)
    #check if anomalies are detected
    if(np.sum(anomalies) == 0):
        print("no anomalies detected in ID: ", ID)
    else:
        print("anomalies detected in ID: ", ID)
        plt.plot(mse, label='MSE', color='blue')
        plt.plot(np.arange(len(mse)), threshold*np.ones(len(mse)), label='Threshold', color='red')
        plt.xlabel('Sample Index')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()
