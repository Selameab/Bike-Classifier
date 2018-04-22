import os
import numpy as np
import cv2
import tensorflow as tf
from model import Model, IMG_SIZE

TEST_IMAGES_PATH = 'bikes_ds/test_images'
CLS_DICT = {0: 'Mountain Bike', 1: 'Road Bike'}

if __name__ == '__main__':
    # Read all images in TEST_IMAGES_PATH
    imgs, imgs_resized, filenames = [], [], []

    for file in os.listdir(TEST_IMAGES_PATH):
        filenames += [file]
        imgs += [cv2.imread(os.path.join(TEST_IMAGES_PATH, file))]
        imgs_resized += [cv2.resize(imgs[-1], tuple(reversed(IMG_SIZE)))]

    # Load model
    model = Model()

    with tf.Session(graph=model.get_graph()) as sess:
        model.compile(sess)

        # Restore model
        model.restore('ckpt/model')

        # Make predictions on all images at once
        predictions = model.predict(imgs_resized)
        predicted_cls = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)

        for i in range(len(predictions)):
            message = '%s, %.1f%% confidence' % (CLS_DICT[predicted_cls[i]], confidence[i] * 100)
            print(filenames[i], message)
            cv2.putText(imgs[i], message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 40, 40), 2, cv2.LINE_AA)
            cv2.imshow(filenames[i], imgs[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
