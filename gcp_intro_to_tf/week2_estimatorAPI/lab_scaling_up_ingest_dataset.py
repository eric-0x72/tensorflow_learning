# 2c. Loading large datasets progressively with the tf.data.Dataset
#
# In this notebook, we continue reading the same small dataset,
# but refactor our ML pipeline in two small, but significant, ways:
#
# 1. Refactor the input to read data from disk progressively.
# 2. Refactor the feature creation so that it is not one-to-one with inputs. </ol>
# The Pandas function in the previous notebook first read the whole data into memory
# -- on a large dataset, this won't be an option.


from google.cloud import bigquery
import tensorflow as tf
import numpy as np
import shutil

# 1. Refactor the input
# Read data created in Lab1a, but this time make it more general, so that we can later handle large datasets.
# We use the Dataset API for this. It ensures that, as data gets delivered to the model in mini-batches,
# it is loaded from disk only when needed.

CSV_COLUMNS = ['fare_amount', 'pickuplon', 'pickuplat', 'dropofflon', 'dropofflat', 'passengers', 'key']
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]


def read_dataset(filename, mode, batch_size=512):
    def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        features.pop('key')  # discard, not a real feature
        label = features.pop('fare_amount')  # remove label from features and store
        return features, label

    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(filename, shuffle=False)
    # Read lines from text files
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    # Parse text lines as comma-separated values (CSV)
    dataset = textlines_dataset.map(decode_csv)

    # Note:
    # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
    # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None  # loop indefinitely
        dataset = dataset.shuffle(buffer_size=10 * batch_size, seed=2)
    else:
        num_epochs = 1  # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset


def get_train_input_fn():
    return read_dataset('./taxi-train.csv', mode=tf.estimator.ModeKeys.TRAIN)


def get_valid_input_fn():
    return read_dataset('./taxi-valid.csv', mode=tf.estimator.ModeKeys.EVAL)


# 2. Refactor the way features are created.
# For now, pass these through (same as previous lab). However, refactoring this way
# will enable us to break the one-to-one relationship between inputs and features.

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]


def add_more_features(feats):
    # Nothing to add (yet!)
    return feats


feature_cols = add_more_features(INPUT_COLUMNS)

# Create and train the model
# Note that we train for num_steps * batch_size examples
tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time

model = tf.estimator.LinearRegressor(feature_columns=feature_cols, model_dir=OUTDIR)
model.train(input_fn=get_train_input_fn, steps=200)

# Evaluate model
# As before, evaluate on the validation data. We'll do the third refactoring
# (to move the evaluation into the training loop) in the next lab.
metrics = model.evaluate(input_fn=get_valid_input_fn, steps=None)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
