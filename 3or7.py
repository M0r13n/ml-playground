#!/usr/bin/env python3

import mnist
import mathutil

# constants

P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'

PIXEL_ASCII_MAP = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def pretty_print(tensor: mathutil.Tensor) -> None:
    # print the image as ASCII grey scale
    for row in tensor.matrix:
        line = ''.join(
            PIXEL_ASCII_MAP[(pixel * len(PIXEL_ASCII_MAP) - 1) // 255] for pixel in row
        )
        print(line)


def main():
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()

    threes, sevens, three_cnt, seven_cnt = mathutil.Tensor(28, 28), mathutil.Tensor(28, 28), 0, 0

    for label, pixels in images:
        # calculate the average 3 & 7
        if label == 3:
            threes = threes + mathutil.Tensor(28, 28, pixels)
            three_cnt += 1
        if label == 7:
            sevens += mathutil.Tensor(28, 28, pixels)
            seven_cnt += 1

    threes = threes * (1 / three_cnt)
    threes.rotate()
    pretty_print(threes)


if __name__ == '__main__':
    main()
