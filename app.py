from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def detect_anomalies():
    data = pd.read_csv('anomaly_detection_data.csv')
    features = ['voltage', 'current', 'temperature']
    data = data[features]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=features)

    time_steps = 10

    def create_sequences(data, time_steps):
        sequences = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
        return np.array(sequences)

    X = create_sequences(data_scaled.values, time_steps)
    y = data_scaled['voltage'][time_steps:].values
    y = y.reshape(-1, 1)

    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(time_steps, len(features))))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=10, validation_split=0.2)

    predictions = model.predict(X)
    anomaly_thresholds = {'voltage': 0.1, 'current': 0.3, 'temperature': 0.1}
    errors = np.abs(predictions - y)

    anomalies = (errors > [anomaly_thresholds['voltage'], anomaly_thresholds['current'], anomaly_thresholds['temperature']])
    anomaly_indices = {'voltage': [], 'current': [], 'temperature': []}

    for i in range(anomalies.shape[0]):
        for j, feature in enumerate(features):
            if anomalies[i, j]:
                anomaly_indices[feature].append(i)

    return anomaly_indices, data_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    anomaly_indices, data_scaled = detect_anomalies()
    return render_template('result.html', anomaly_indices=anomaly_indices, data_scaled=data_scaled.to_dict('records'))

@app.route('/get_chart_data')
def get_chart_data():
    anomaly_indices, data_scaled = detect_anomalies()
    data = {
        'indices': list(data_scaled.index),
        'voltage': list(data_scaled['voltage']),
        'current': list(data_scaled['current']),
        'temperature': list(data_scaled['temperature']),
        'anomalies_voltage': anomaly_indices['voltage'],
        'anomalies_current': anomaly_indices['current'],
        'anomalies_temperature': anomaly_indices['temperature']
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
