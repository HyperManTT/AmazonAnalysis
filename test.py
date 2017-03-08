import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        'asin': np.array(["A", "B", "C", "B", "A", "D"]),
        'B': np.array([4] * 6, dtype='int32'),
        'C': np.array([5] * 6, dtype='int32'),
        'D': np.array([6] * 6, dtype='int32')

    }
)

print df['asin'].value_counts()
print len(df.index)

print df['asin'].value_counts()/len(df.index)