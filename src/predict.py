import joblib

model = joblib.load("KNNeighborsML/src/KNN.pkl")
prediction = model.predict([[-1.781797,-1.490046]])
print(prediction)
