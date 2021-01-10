from tensorflow import keras
from sklearn import preprocessing
import numpy as np
import pandas as pd
from scipy import stats

normalization_max = 66.615074
normalization_min = -78.47761
labels = ['A',
          'B',
          'C',
          'D',
          'E']
activities = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing"
}
TIME_PERIODS = 200
STEP_DISTANCE = 200
test_segments = []
test_labels = []


def create_segments_and_labels(dff, time_steps, step):
    print('Creating Segments...')
    # x, y, z acceleration as features
    n_features = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    accel_data = dff[['watch-accel-x', 'watch-accel-y', 'watch-accel-z']]
    for i in range(0, len(dff) - time_steps, step):
        values = accel_data.iloc[i:(i + time_steps)].values
        # Retrieve the most often used label in this segment
        label = stats.mode(dff['activity'][i: i + time_steps])[0][0]
        segments.append(values)
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)
    return reshaped_segments, labels


def normalise_data(df):
    print('Normalising Data...')
    ndf = df.copy()
    ndf['watch-accel-x'] = (df['watch-accel-x'] - normalization_min) / (normalization_max - normalization_min)
    ndf['watch-accel-y'] = (df['watch-accel-y'] - normalization_min) / (normalization_max - normalization_min)
    ndf['watch-accel-z'] = (df['watch-accel-z'] - normalization_min) / (normalization_max - normalization_min)
    ndf = ndf.round({'watch-accel-x': 4, 'watch-accel-y': 4, 'watch-accel-z': 4})
    return ndf


def read_data():
    global test_segments, test_labels
    column_names = ['activity',
                    'timestamp',
                    'watch-accel-x',
                    'watch-accel-y',
                    'watch-accel-z',
                    'phone-accel-x',
                    'phone-accel-y',
                    'phone-accel-z']
    print('Reading Data...')
    df = pd.read_csv('data_compact.csv', header=None, names=column_names)
    ndf = normalise_data(df)
    test_segments, test_labels = create_segments_and_labels(ndf, TIME_PERIODS, STEP_DISTANCE)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_data()
    model = keras.models.load_model("watch")
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    while True:
        index = int(input("Enter an index:"))
        values = test_segments[index]
        input_data = values.reshape(1, 200 * 3)
        encoded_prediction = np.argmax(model.predict(input_data), axis=1)
        decoded_prediction = le.inverse_transform(encoded_prediction)[0]
        print("Activity: " + activities[test_labels[index]])
        print("Predicted Activity: " + activities[decoded_prediction])
