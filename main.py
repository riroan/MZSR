import cv2
import numpy as np
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
import argparse

def getModel(input_shape):
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=input_shape, activation="relu", padding="same"))
    model.add(UpSampling2D())
    for _ in range(6):
        model.add(Conv2D(64, 3, activation="relu", padding="same"))

    model.add(Conv2D(3, 3, padding="same"))
    model.compile(optimizer="adam", loss="mae")
    model.summary()
    return model


def downSampling(img):
    img = np.array(img, dtype=np.float64)
    h, w, c = img.shape
    dst = cv2.resize(img, dsize=(w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    print(dst.shape)
    gaussian_noise = np.random.normal(0, 1, (h // 2, w // 2, c))
    dst += gaussian_noise
    dst = np.clip(dst, 0, 255)
    dst = np.array(dst, dtype=np.uint8)

    return dst


def MZSR(I):
    I_son = downSampling(I)
    model = getModel(I_son.shape)

    # Image Normalization
    X = I_son / 127.5 - 1
    Y = I / 127.5 - 1

    # zero-shot training
    model.fit(np.array([X], dtype=np.float32), np.array([Y], dtype=np.float32), epochs=100)
    model.save_weights("w.h5")

    model_de = getModel(I.shape)
    model_de.load_weights("w.h5")

    HR = np.uint((model.predict(np.array([Y]))[0] + 1) * 127.5)
    HR = np.array(HR, dtype=np.uint8)

    cv2.imshow("result", HR)
    cv2.imwrite("result.png", HR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    I = cv2.imread(args.path)
    MZSR(I)
