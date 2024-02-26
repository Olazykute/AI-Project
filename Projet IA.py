import numpy as np
import matplotlib.pyplot as plt
import polars as pl

data_0 = pl.read_csv("Features_by_window_size/sero_features_4.csv")
print(data_0)

data_1=[]
data_2=[]
data_3=[]
data_4=[]

data = {
    0: data_0,
    1: data_1,
    2: data_2,
    3: data_3,
    4: data_4
}

i = 0
for j in range(1,4):
    while data_0[i,0] == j:
        i+=1
    data[j]= data_0[0:i,:]
    print(data[j])
    








"""
target = data_1[:,"Target"]
x = np.linspace(0,target.len(),target.len())
y = data_1[:,"AccMeanX"]
y2 = data_1[:,"AccMeanY"]
y3 = data_1[:,"AccMeanZ"]

y4 = data_1[:,"GyroMeanX"]
y5 = data_1[:,"GyroMeanY"]
y6 = data_1[:,"GyroMeanZ"]


plt.figure()
plt.plot(x,y,color="red")
plt.plot(x,y2,color="blue")
plt.plot(x,y3,color="green")

plt.figure()
plt.plot(x,y4,color="red")
plt.plot(x,y5,color="blue")
plt.plot(x,y6,color="green")

plt.show()
"""
