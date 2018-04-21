import tensorflow as tf

IMG_SIZE = (536, 334)
LEARNING_RATE = 1e-4


class Model:
    def __init__(self):
        self.__sess = None
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            # Input Tensors
            x = tf.placeholder(dtype=tf.float32, shape=(None,) + IMG_SIZE + (3,), name='x')
            y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='y')

            # Layers
            layers = [
                ('CONV1_1', tf.layers.Conv2D(name='CONV1_1', kernel_size=3, filters=16, padding='same',
                                             activation=tf.nn.relu)),
                ('CONV1_2', tf.layers.Conv2D(name='CONV1_2', kernel_size=3, filters=16, padding='same',
                                             activation=tf.nn.relu)),
                ('POOL1', tf.layers.MaxPooling2D(name='POOL1', pool_size=2, strides=2)),

                ('CONV2_1', tf.layers.Conv2D(name='CONV2_1', kernel_size=3, filters=32, padding='same',
                                             activation=tf.nn.relu)),
                ('CONV2_2', tf.layers.Conv2D(name='CONV2_2', kernel_size=3, filters=32, padding='same',
                                             activation=tf.nn.relu)),
                ('POOL2', tf.layers.MaxPooling2D(name='POOL2', pool_size=2, strides=2)),

                ('CONV3_1', tf.layers.Conv2D(name='CONV3_1', kernel_size=3, filters=16, strides=2, padding='same',
                                             activation=tf.nn.relu)),
                ('POOL3', tf.layers.MaxPooling2D(name='POOL2', pool_size=2, strides=2)),

                ('FLATTTEN', tf.layers.Flatten()),

                ('FC1', tf.layers.Dense(name='FC1', units=32, activation=tf.nn.relu)),
                ('FC2', tf.layers.Dense(name='FC2', units=16, activation=tf.nn.relu)),
                ('FC3', tf.layers.Dense(name='FC3', units=2, activation=None))
            ]

            # Connect layers
            self.__tensors = {'INPUT': x}
            net = x
            for key, layer in layers:
                net = layer(net)
                self.__tensors[key] = net

            y_pred = tf.nn.softmax(net)
            y_pred_cls = tf.argmax(y_pred, axis=1)
            y_cls = tf.argmax(y, axis=1)

            # Optimization
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_cls, logits=net))
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_cls, y_pred_cls), tf.float32))
            saver = tf.train.Saver()

            self.__x = x
            self.__y = y
            self.__layers = layers
            self.__optimizer = optimizer
            self.__y_pred = y_pred
            self.__loss = loss
            self.__accuracy = accuracy
            self.__saver = saver

    def get_graph(self):
        return self.__graph

    def get_all_layers(self):
        return self.__layers

    def compile(self, sess):
        self.__sess = sess
        with self.get_graph().as_default():
            self.__sess.run(tf.global_variables_initializer())

    def train(self, x_batch, y_batch):
        self.__sess.run([self.__optimizer], feed_dict={self.__x: x_batch, self.__y: y_batch})

    def predict(self, x_batch):
        return self.__sess.run(self.__y_pred, feed_dict={self.__x: x_batch})

    def get_accuracy(self, x_batch, y_batch):
        return self.__sess.run(self.__accuracy, feed_dict={self.__x: x_batch, self.__y: y_batch})

    def get_loss(self, x_batch, y_batch):
        return self.__sess.run(self.__loss, feed_dict={self.__x: x_batch, self.__y: y_batch})

    def save(self, path):
        self.__saver.save(self.__sess, path)

    def restore(self, path):
        self.__saver.restore(self.__sess, path)
