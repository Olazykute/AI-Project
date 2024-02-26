import numpy as np
import matplotlib.pyplot as plt
import polars as pol

data = pol.read_csv("sero_features_4.csv")
print(data)
#print(data[:,0])

target = data[:,0]
x = np.linspace(0,target.len(),target.len())
y = data[:,"AccMeanX"]
y2 = data[:,"AccMeanY"]
y3 = data[:,"AccMeanZ"]

y4 = data[:,"GyroMeanX"]
y5 = data[:,"GyroMeanY"]
y6 = data[:,"GyroMeanZ"]

plt.figure()
plt.plot(x,y,color="red")
plt.plot(x,y2,color="blue")
plt.plot(x,y3,color="green")

plt.figure()
plt.plot(x,y4,color="red")
plt.plot(x,y5,color="blue")
plt.plot(x,y6,color="green")

plt.show()