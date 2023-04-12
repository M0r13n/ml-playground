import dataclasses
import itertools
import pathlib
import gzip
import struct
import typing

# constants

P_TRAINING_LABEL_FILE = 'res/gzip/emnist-digits-train-labels-idx1-ubyte.gz'
P_TRAINING_IMG_FILE = 'res/gzip/emnist-digits-train-images-idx3-ubyte.gz'

PIXEL_ASCII_MAP = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


# utils

def chunkify(fd):
    while chunk:= fd.read(4096):
        for byte in chunk:
            yield byte

def row_wise(stream, n_rows):
    buffer = []
    for i, pixel in enumerate(stream, 1):
        buffer.append(pixel)
        if i % n_rows == 0:
            yield buffer
            buffer = []

def mat_add(a, b):
    res = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[0])):
            row.append(a[i][j]+b[i][j])
        res.append(row)
    return res

def scalar_mult(a, func):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] = func(a[i][j])
    return a


@dataclasses.dataclass
class Image:
    label: str
    rows: typing.List[typing.List[int]]

    def pretty_print(self):
        # print the image as ASCII grey scale
        for row in self.rows:
            line = ''.join(PIXEL_ASCII_MAP[(pixel * len(PIXEL_ASCII_MAP) -1) // 255] for pixel in row)
            print(line)
        print('==>', self.label)


    def rotate(self):
        self.rows = list(zip(*self.rows))

class MNISTReader:
    IDX1_MAGIC_NUMBER = 2049
    IDX3_MAGIC_NUMBER = 2051

    @dataclasses.dataclass
    class IDX1Header:
        magic_number: int
        num_items: int

    @dataclasses.dataclass
    class IDX3Header:
        magic_number: int
        num_items: int
        num_rows: int
        num_cols: int


    def __init__(self, label_file: str, image_file:str) -> None:
        # make those strings to paths in order to easily work with them
        self.label_file = pathlib.Path(label_file)
        self.image_file = pathlib.Path(image_file)

        # make sure that the files actually exist
        assert self.label_file.exists() and self.label_file.is_file()
        assert self.image_file.exists() and self.label_file.is_file()

    def read(self):
        with gzip.open(self.label_file, 'r') as lfd:
            with gzip.open(self.image_file, 'r') as ifd:
                # read header of both files
                i1h = self.IDX1Header(*struct.unpack('>ii', lfd.read(8)))
                i3h = self.IDX3Header(*struct.unpack('>iiii', ifd.read(16)))

                # make sure that both files define the same number of items
                assert i1h.num_items == i3h.num_items

                # make sure that the magic numbers are correct
                assert i1h.magic_number == self.IDX1_MAGIC_NUMBER
                assert i3h.magic_number == self.IDX3_MAGIC_NUMBER
            
                # lazily read labels & pixels
                labels = chunkify(lfd)
                pixels = chunkify(ifd)

                # read pixels row-wise
                rows = []
                for i, row in enumerate(row_wise(pixels, i3h.num_cols), 1):
                    rows.append(row)
                    if i % i3h.num_rows == 0:
                        yield Image(next(labels), rows)
                        rows = []


def main():
    reader = MNISTReader(P_TRAINING_LABEL_FILE, P_TRAINING_IMG_FILE)
    images = reader.read()

    threes =  [[0]*28 for _ in range(28)]
    three_cnt = 0
    sevens =  [[0]*28 for _ in range(28)]

    for img in images:
        # calculate the average 3 & 7
        if img.label == 3:
            threes = mat_add(threes, img.rows)
            three_cnt += 1
        if img.label == 7:
            sevens = mat_add(sevens, img.rows)

    scalar_mult(threes,  lambda x: x//three_cnt)
    threes = Image(3, threes)
    threes.rotate()
    threes.pretty_print()

if __name__ == '__main__':
    main()
