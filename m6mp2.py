import pandas as pd

# STEP 1 Load document

file    = ('dataset/titanic.csv')
data    = pd.read_csv(file) 
print(data.shape)