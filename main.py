import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pth
from datetime import datetime


path = 'Acuity_LS_00833_20250226_102627.csv'
# path = pth.Path(path)
cols = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)',
        'RH (%)','Surface Temp (C)' ,'Cond Lo Freq (S)',
        'Cond Hi Freq (S)','Galv Corr 1 (A)','Galv Corr 2 (A)',
        'Tot Cond Lo Freq (C/V)','Tot Cond Hi Freq (C/V)',
        'Tot Galv Corr 1 (C)','Tot Galv Corr 2 (C)']



data = pd.read_csv(path, header = None)
data = data.iloc[1:, :]
data.columns = cols


data['Unix Time (s)'] = pd.to_datetime(data['Unix Time (s)'],unit='s')
data = data.rename(columns={'Unix Time (s)': 'Date-Time'})

data.iloc[:, 1:] = data.iloc[:, 1:].astype(dtype = float)

data = data.iloc[:, 1:]
print(data.columns)

x = 'Air Temp (C)'
y = 'RH (%)'

plt.plot(data[x], data[y])
plt.show()