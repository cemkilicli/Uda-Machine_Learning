from sklearn.preprocessing import MinMaxScaler
import numpy as np

weights = np.array([[115.],[140.],[175.]])

scaler = MinMaxScaler()
rescaled_weight =scaler.fit_transform(weights)

print rescaled_weight
