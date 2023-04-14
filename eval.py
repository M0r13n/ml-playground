import gzip
import time
import typing


P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'


def read_in_chunks(fd: gzip.GzipFile) -> typing.Generator[int, None, None]:
    while chunk := fd.read(4096):
        for byte in chunk:
            yield byte


start = time.time()
with gzip.open(P_TRAINING_LABEL_FILE, 'r') as lfd:
    with gzip.open(P_TRAINING_IMG_FILE, 'r') as ifd:
        # lazily read labels & pixels
        labels = read_in_chunks(lfd)
        pixels = read_in_chunks(ifd)

        # read pixels row-wise
        rows = []  # type:ignore
        for i, pixel in enumerate(pixels, 1):
            if i % (28 * 28) == 0:
                next(labels)

print(time.time() - start)
