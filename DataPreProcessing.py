import pandas as pd

testA = pd.read_csv("CSV/Restaurants_Test_Data_PhaseA.csv")
testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainA = pd.read_csv("CSV/Restaurants_Train.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

print(testA.head(5))
print(testB.head(5))
print(trainA.head(5))
print(trainB.head(5))
