def featureScaling(arr):
    scaled_features = []
    x_max = max(arr)
    x_min = min(arr)

    for i in range(0, len(arr)):
        x_1 = float(arr[i]-x_min)
        x_2 = float(x_max-x_min)
        x_new = x_1/x_2
        scaled_features.append(x_new)
    return scaled_features

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
