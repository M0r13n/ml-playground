"""Use back propagation to train a simple model to differentiate between threes and sevens"""
import pretty
import mnist
import mathutil

P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'

P_VALIDATION_LABEL_FILE = 'res/gzip/emnist-digits-test-labels-idx1-ubyte.gz'
P_VALIDATION_IMG_FILE = 'res/gzip/emnist-digits-test-images-idx3-ubyte.gz'


def test(bias, weights):
    reader = mnist.MNISTReader(P_VALIDATION_LABEL_FILE, P_VALIDATION_IMG_FILE)
    images = reader.read()
    correct, total, = 0, 0

    # iterate over all test again and compute: image∘w+b
    for i, (label, pixels) in enumerate(images, 1):
        if label == 3:
            r = mathutil.Tensor(28, 28, pixels).hadamard_product(weights)
            if (r.sum() + bias) < 0:
                # negative value => number is most likely a three
                correct += 1
            total += 1
        if label == 7:
            r = mathutil.Tensor(28, 28, pixels).hadamard_product(weights)
            if (r.sum() + bias) >= 0:
                # positive value => number is most likely a seven
                correct += 1
            total += 1
    accuracy = 100.0 * correct / total
    print(accuracy, correct, total)
    return accuracy


def main():
    reader = mnist.MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()

    # initialize a random weights matrix.
    weights = mathutil.Tensor(28, 28)
    weights.randomize(-10, 10)

    # create a identity matrix
    learning_rate = mathutil.Tensor.identity(28, 28) * 0.01
    bias = 0
    t = 0

    test(bias, weights)

    for label, pixels in images:
        if label not in (3, 7):
            continue

        t += 1
        img = mathutil.Tensor(28, 28, pixels)

        # if t % 20000 == 0:
        #    test(bias, weights)

        # compute the prediction value
        r = img.hadamard_product(weights).sum() + bias

        if label == 3:
            if r > 5000:
                # apply the learning rate to the image
                weights = weights - img * learning_rate
                bias -= 0.01
        else:
            if r < -5000:
                weights = weights + img * learning_rate
                bias += 0.01

    test(bias, weights)
    weights.rotate()
    pretty.pretty_print(weights)


if __name__ == '__main__':
    main()
