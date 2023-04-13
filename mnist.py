import dataclasses
import gzip
import pathlib
import struct
import typing


def read_in_chunks(fd: gzip.GzipFile) -> typing.Generator[int, None, None]:
    while chunk := fd.read(4096):
        for byte in chunk:
            yield byte


def row_wise(
        stream: typing.Generator[int, None, None], n_rows: int
) -> typing.Generator[typing.List[int], None, None]:
    buffer = []
    for i, pixel in enumerate(stream, 1):
        buffer.append(pixel)
        if i % n_rows == 0:
            yield buffer
            buffer = []


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

    def __init__(self, label_file: str, image_file: str) -> None:
        # make those strings to paths in order to easily work with them
        self.label_file = pathlib.Path(label_file)
        self.image_file = pathlib.Path(image_file)

        # make sure that the files actually exist
        assert self.label_file.exists() and self.label_file.is_file()
        assert self.image_file.exists() and self.label_file.is_file()

    def read(self) -> typing.Generator[typing.Tuple[int, typing.List[typing.List[int]]], None, None]:
        """Read the MNIST files and lazily yield images as tuples.
        Each tuple has two values: (label, pixels), where pixels is a NxN
        2d array of 8 bit integer values 0-255.
        """
        with gzip.open(self.label_file, 'r') as lfd:
            with gzip.open(self.image_file, 'r') as ifd:
                # read header of both files
                i1h = self.IDX1Header(*struct.unpack('>ii', lfd.read(8)))
                i3h = self.IDX3Header(*struct.unpack('>iiii', ifd.read(16)))

                # make sure that both files define the same number of items
                assert i1h.num_items == i3h.num_items, "idx1 and idx3 define unequal number of items"

                # make sure that the magic numbers are correct
                assert i1h.magic_number == self.IDX1_MAGIC_NUMBER, "idx1 invalid magic number"
                assert i3h.magic_number == self.IDX3_MAGIC_NUMBER, "idx3 invalid magic number"

                # lazily read labels & pixels
                labels = read_in_chunks(lfd)
                pixels = read_in_chunks(ifd)

                # read pixels row-wise
                rows = []
                for i, row in enumerate(row_wise(pixels, i3h.num_cols), 1):
                    rows.append(row)
                    if i % i3h.num_rows == 0:
                        yield next(labels), rows
                        rows = []
