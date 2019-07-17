# Distributed training and monitoring

'''
In this notebook, we refactor to call train_and_evaluate instead of hand-coding our ML pipeline.
This allows us to carry out evaluation as part of our training loop instead of as a separate step.
It also adds in failure-handling that is necessary for distributed training capabilities.
We also use TensorBoard to monitor the training.
'''

from google.cloud import bigquery
import tensorflow as tf
import numpy as np
import shutil
from google.datalab.ml import TensorBoard

CSV_COLUMNS = ['fare_amount', 'pickuplon', 'pickuplat', 'dropofflon', 'dropofflat', 'passengers', 'key']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]


def read_dataset(filename, mode, batch_size=512):
    def decode_csv(value_column):
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        # No need to features.pop('key') since it is not specified in the INPUT_COLUMNS.
        # The key passes through the graph unused.
        return features, label

    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(filename)
    # Read lines from text files
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    # Parse text lines as comma-separated values (CSV)
    dataset = textlines_dataset.map(decode_csv)

    # Note:
    # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
    # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None  # indefinitely
        dataset = dataset.shuffle(buffer_size=10 * batch_size)
    else:
        num_epochs = 1  # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset


# Create features out of input data
# For now, pass these through. (same as previous lab)
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


# Serving input function

# Defines the expected shape of the JSON feed that the model will receive once deployed behind a REST API in production
def serving_input_fn():
    json_feature_placeholders = {
        'pickuplon': tf.placeholder(tf.float32, [None]),
        'pickuplat': tf.placeholder(tf.float32, [None]),
        'dropofflat': tf.placeholder(tf.float32, [None]),
        'dropofflon': tf.placeholder(tf.float32, [None]),
        'passengers': tf.placeholder(tf.float32, [None]),
    }
    # You can transforma data here from the input format to the format expected by your model.
    features = json_feature_placeholders  # no transformation needed
    return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)


# tf.estimator.train_and_evaluate
def train_and_evaluate(output_dir, num_train_steps):
    estimator = tf.estimator.LinearRegressor(
        model_dir=output_dir,
        feature_columns=feature_cols)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: read_dataset('./taxi-train.csv', mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=num_train_steps)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: read_dataset('./taxi-valid.csv', mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=1,  # start evaluating after N seconds
        throttle_secs=10,  # evaluate every N seconds
        exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Monitoring with TensorBoard
# Start the TensorBoard by opening up a new Launcher (File > New Launcher) and selecting TensorBoard.
OUTDIR = './taxi_trained'
# TensorBoard().start(OUTDIR)

# Run training
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file
train_and_evaluate(OUTDIR, num_train_steps=500)
