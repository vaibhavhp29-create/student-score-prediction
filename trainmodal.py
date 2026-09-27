import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
}

df = pd.DataFrame(data)

# Input and output
X = df[["hours"]]
y = df["score"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")