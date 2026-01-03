import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\pythonsample\weather-weka.csv")
df.columns = df.columns.str.strip()
#df = df.dropna(subset=['Weather Condition'])
X = df[['temperature', 'humidity']].values
y_raw = df['outlook'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
a = int(input("Enter Temperature: "))
b = int(input("Enter Humidity: "))
new_day = np.array([[a, b]])
predicted_label = knn.predict(new_day)
predicted_weather = label_encoder.inverse_transform(predicted_label)
print("Predicted Weather:", predicted_weather[0])
labels = label_encoder.classes_
colors = plt.cm.tab10(range(len(labels)))
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(
        X[y == i, 0],
        X[y == i, 1],
        c=[colors[i]],
        label=label,
        edgecolor='k',
        s=100
    )
plt.scatter(
    new_day[0, 0],
    new_day[0, 1],
    c='red',
    marker='*',
    s=250,
    label=f'New Day (Predicted: {predicted_weather[0]})'
)
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Weather Prediction using KNN")
plt.legend()
plt.show()
