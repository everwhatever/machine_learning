import numpy as np
import pandas as pd
arr=np.arange(25).reshape(5,5)
df=pd.DataFrame(arr,['A','B','C','D','E'],['a','b','c','d','e'])
print(df[df['b']>6][['c','d']])
