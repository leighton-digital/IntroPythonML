import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

def heatmap(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False, square=True, mask=mask, annot_kws={"fontsize": 4})


data = pd.read_csv('Combined.csv')
cols = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB',
               'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6', 'ER7',
               'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']
data2 = data.filter(cols)
data2 = pd.DataFrame(RobustScaler().fit_transform(np.array(data2)), columns=cols)
data2['%top'] = data['%top']
data2 = data2.dropna(axis=0)
print(data2)
plt.figure(figsize=(8,8))
heatmap(data2)
plt.show()