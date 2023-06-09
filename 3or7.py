#!/usr/bin/env python3
import time
import colorama
import colorama.initialise

import pretty
import mnist
import mathutil

# constants

P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'

P_VALIDATION_LABEL_FILE = 'res/gzip/emnist-digits-test-labels-idx1-ubyte.gz'
P_VALIDATION_IMG_FILE = 'res/gzip/emnist-digits-test-images-idx3-ubyte.gz'


def main():
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()

    threes, sevens, three_cnt, seven_cnt = mathutil.Tensor(28, 28), mathutil.Tensor(28, 28), 0, 0

    start = time.time()
    for label, pixels in images:
        if label == 3:
            three = mathutil.Tensor(28, 28, pixels)
            threes += three
            three_cnt += 1
        if label == 7:
            seven = mathutil.Tensor(28, 28, pixels)
            sevens += seven
            seven_cnt += 1

    # compute the difference between sevens and threes
    delta = (sevens - threes) / (three_cnt + seven_cnt)
    print("Delta computation took:", time.time() - start)

    # compute the bias
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()
    mean3, mean7, start = 0, 0, time.time()

    for label, pixels in images:
        if label not in (3, 7):
            continue
        s = mathutil.Tensor(28, 28, pixels).hadamard_product(delta).sum()

        if label == 3:
            mean3 += s
        if label == 7:
            mean7 += s

    # compute bias: the middle of the means
    bias = -(mean7 / seven_cnt + mean3 / three_cnt) / 2
    print("Bias computation took:", time.time() - start)
    print('Bias:', bias)

    # create a new reader
    reader = mnist.MNISTReader(P_VALIDATION_LABEL_FILE, P_VALIDATION_IMG_FILE)
    images = reader.read()
    correct, total, start = 0, 0, time.time()

    # iterate over all test again and compute: image∘w+b
    for i, (label, pixels) in enumerate(images, 1):
        if label == 3:
            r = mathutil.Tensor(28, 28, pixels).hadamard_product(delta)
            if (r.sum() + bias) < 0:
                # negative value => number is most likely a three
                correct += 1
            total += 1
        if label == 7:
            r = mathutil.Tensor(28, 28, pixels).hadamard_product(delta)
            if (r.sum() + bias) >= 0:
                # positive value => number is most likely a seven
                correct += 1
            total += 1

    print("Validation took:", time.time() - start)
    print(f'Looked at 240.000 training images and {i} validation images.')
    print(f'Accuracy: {correct / total :.0%}')

    delta.rotate()
    pretty.pretty_print(delta)


if __name__ == '__main__':
    colorama.initialise.init()
    main()
