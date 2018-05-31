import pandas as pd
import numpy as np

ACTIVITIES = {
    0: 'NOSTAIRS',
    1: 'STAIRS'
}

def confusion_matrix(Y_true, Y_pred, activities = ACTIVITIES):
    Y_true = pd.Series([activities[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([activities[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])
