import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

DATADIR = 'WISDM_RAW'

FILENAME = 'WISDM_RAW/WISDM_raw.txt'

TRAINING_DATA = 0.8

N_TIME_STEPS = 200

N_FEATURES = 3

step = 20

RANDOM_SEED = 42

LABELS = {
    'Walking' : 0,
    'Jogging': 1,
    'Stairs': 2,
    'Sitting': 3,
    'Standing': 4,
    'LyingDown' : 5
}

def one_hot(num):
    return [int(i == num) for i in range(len(LABELS))]

def load_data():
    columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(FILENAME, header = None, names = columns)

    segments = []
    labels = []

    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = df['x-axis'].values[i: i + N_TIME_STEPS]
        ys = df['y-axis'].values[i: i + N_TIME_STEPS]
        zs = df['z-axis'].values[i: i + N_TIME_STEPS]
        try:
            label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
            if label not in LABELS:
                continue
        except:
            #print('error in line ' + str(i) + '. skipping...')
            continue
        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    print(reshaped_segments.shape)
    print(labels.shape)


    X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_data()
    print('data loaded.')