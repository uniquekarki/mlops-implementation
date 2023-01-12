import mlflow
import numpy as np
logged_model = 'runs:/212546f8b757415f8639915525e80484/model-artifact'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


data = [["0.00632",
     "18.00",
     "2.310",
     "0",
     "0.5380",
     "6.5750",
     "65.20",
     "4.090",
     "1",
     "296.0",
     "15.30",
     "396.90",
     "4.98",
]]

truth = "24.00",
data_np = np.array(data)
# Predict on a Pandas DataFrame.
prediction = loaded_model.predict(data)

print("\n----------------TEST RESULTS----------------\n")
print('Actual Value:',truth[0])
print('Predicted Value:',prediction[0])
print("\n--------------------------------------------\n")
