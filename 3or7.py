#!/usr/bin/env python3
import colorama
import colorama.initialise

import mnist
import mathutil

# constants

P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'

P_VALIDATION_LABEL_FILE = 'res/gzip/emnist-digits-test-labels-idx1-ubyte.gz'
P_VALIDATION_IMG_FILE = 'res/gzip/emnist-digits-test-images-idx3-ubyte.gz'

PIXEL_ASCII_MAP = " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@ "


def pretty_print(tensor: mathutil.Tensor) -> None:
    # print the image as ASCII grey scale
    for row in tensor.matrix:
        line = ''
        for pixel in row:
            char = PIXEL_ASCII_MAP[int((pixel * len(PIXEL_ASCII_MAP) - 1) // 255)]
            if pixel < 0:
                line += f'{colorama.Fore.RED}{char}{colorama.Style.RESET_ALL}'
            else:
                line += char

        print(line)


def main():
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()

    threes, sevens, three_cnt, seven_cnt = mathutil.Tensor(28, 28), mathutil.Tensor(28, 28), 0, 0

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

    # compute the bias
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()
    mean3, mean7 = 0, 0

    for label, pixels in images:
        s = mathutil.Tensor(28, 28, pixels).hadamard_product(delta).sum()
        if label == 3:
            mean3 += s
        if label == 7:
            mean7 += s

    # compute bias: the middle of the means
    bias = -(mean7 / seven_cnt + mean3 / three_cnt) / 2

    # create a new reader
    reader = mnist.MNISTReader(P_VALIDATION_LABEL_FILE, P_VALIDATION_IMG_FILE)
    images = reader.read()
    correct, total = 0, 0

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

    print(f'Looking at 240.000 training images and {i} validation images.')
    print(f'Accuracy: {correct / total :.0%}')


if __name__ == '__main__':
    colorama.initialise.init()
    main()
