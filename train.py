import os
import numpy as np
import tensorflow as tf
import cv2
from model import Model, IMG_SIZE
from sklearn.model_selection import train_test_split

EPOCHS = 100
TRAINING_IMAGES_PATH = 'bikes_ds/training_images'


def load_dataset(path):
    x, y = [], []

    # Iterate through each folder in path
    cls_count = 0
    for sub_class in os.listdir(path):
        for file in os.listdir(os.path.join(path, sub_class)):
            y += [cls_count]
            x += [cv2.resize(cv2.imread(os.path.join(path, sub_class, file)), tuple(reversed(IMG_SIZE)))]
        cls_count += 1

    # Convert to numpy array
    x = np.array(x)
    y = np.eye(cls_count)[y]  # One hot encode

    # Split into training and validation set
    return train_test_split(x, y, test_size=0.8, random_state=16063)


if __name__ == '__main__':
    # Load dataset
    x_train, x_val, y_train, y_val = load_dataset(TRAINING_IMAGES_PATH)
    print("%d training and %d validation images loaded" % (y_train.shape[0], y_val.shape[0]))

    # Create Model
    model = Model()
    with tf.Session(graph=model.get_graph()) as sess:
        model.compile(sess)
        print("Training started...")
        for i in range(EPOCHS):
            model.train(x_train, y_train)
            if i % 10 == 0:
                training_loss = model.get_loss(x_train, y_train)
                val_loss = model.get_loss(x_val, y_val)
                print("Epoch %i   Training Loss = %f   Validation Loss = %f" % (i, training_loss, val_loss))

                # Early stopping
                if training_loss < 0.01 and val_loss < 0.02:
                    print("Breaking at epoch %i" % i)
                    break

        print("*" * 60)
        print("Training Acc = %.2f   Validation Acc = %.2f" % (
            model.get_accuracy(x_train, y_train), model.get_accuracy(x_val, y_val)))

        model.save('ckpt/model')
