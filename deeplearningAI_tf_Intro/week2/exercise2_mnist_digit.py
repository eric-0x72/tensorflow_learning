import tensorflow as tf

'''
Exercise 2
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. 
There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs 

Some notes:
It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?
'''


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.90:
            print("\nReached 0.90 accuracy so cancelling training!")
            self.model.stop_training = True


mnist = tf.keras.datasets.mnist
callbacks = myCallback()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # must normalise the data !!
print('x train shape is: ', x_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # input_shape=(28,28)
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 digits
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])  # add callbacks
