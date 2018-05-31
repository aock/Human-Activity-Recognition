import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

DATADIR = 'WISDM_RAW'

FILENAME = 'WISDM_RAW/WISDM_raw.txt'

LABELS = {
    'NoStairs' : 0,
    'Stairs' : 1
}

LABEL_COUNTER = {
    0 : 0.0,
    1 : 0.0
}

def one_hot(num):
    return [int(i == num) for i in range(len(LABELS))]

def load_data(step=20,
              random_seed=42,
              n_features = 3,
              test_size = 0.2,
              n_time_steps = 200,
              filename='WISDM_RAW/WISDM_raw.txt'):
    print("read data from %s ..." % filename)
    columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(filename, header = None, names = columns)

    segments = []
    labels = []

    print("number of samples found: %d" % len(df))
    print("reorganize data...")


    for i in range(0, len(df) - n_time_steps, step):
        xs = df['x-axis'].values[i: i + n_time_steps]
        ys = df['y-axis'].values[i: i + n_time_steps]
        zs = df['z-axis'].values[i: i + n_time_steps]
        try:
            label = stats.mode(df['activity'][i: i + n_time_steps])[0][0]
            if label not in LABELS:
                continue
            else:
                num = LABELS[label]
                LABEL_COUNTER[num] += 1.0
                label = one_hot(num)
        except:
            #print('error in line ' + str(i) + '. skipping...')
            continue
        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    labels = np.asarray(labels, dtype = np.float32)
    print(reshaped_segments.shape)
    print(labels.shape)


    X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=test_size, random_state=random_seed)

    return X_train, X_test, y_train, y_test, LABEL_COUNTER

if __name__ == "__main__":
    load_data()
    print('data loaded.')
