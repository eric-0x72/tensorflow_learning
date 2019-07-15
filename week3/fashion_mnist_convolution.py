import tensorflow as tf

print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # only check if epoch ends
        if logs.get('acc') > 0.8:
            # print(logs) {'loss': 0.47559950805107754, 'acc': 0.82945}
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 28*28 size, 1 channel
    tf.keras.layers.MaxPooling2D(2, 2),  # pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # conv
    tf.keras.layers.MaxPooling2D(2, 2),  # pooling

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)
print(test_loss)
