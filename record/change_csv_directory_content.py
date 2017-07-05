#from pandas.io.parsers import read_csv
import os
import numpy as np
import pandas as pd
import numpy.core.defchararray as np_f

filename = 'driving_log.csv'
#df = pd.read_csv(filename, float_precision='high', header=None)
df = pd.read_csv(filename, float_precision='high')
df.columns = ["Center Image", "Left Image", "Right Image", "Steering", "Throttle", "Brake", "Speed"]
#print(df.index)
#print(df.columns)
data = df.values
# parse string "os folder"

# Step1. find current pwd
cwd = os.getcwd()
# Step2. find the old Path in data[0, 0]
index = data[0, 0].find('/IMG')
old_folder_path = data[0, 0][:index]

# numpy.core.defchararray.replace function

new_data = []
for row in data:
    new_row = np_f.replace(list(row), old_folder_path, cwd)
    new_data.append(new_row)

new_data = np.array(new_data)

new_df = pd.DataFrame(new_data, columns=df.columns)

new_df.to_csv(filename, index=False)
