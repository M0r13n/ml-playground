import colorama
import png
import mathutil


PIXEL_ASCII_MAP = " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@ "


def pretty_print(tensor: mathutil.Tensor) -> None:
    # compute upper and lower bounds for normalization
    lmin, lmax = 0.0, 0.0
    for row in tensor.matrix:
        for pixel in row:
            lmin = min(lmin, pixel)
            lmax = max(lmax, pixel)

    for row in tensor.matrix:
        line = ''
        for val in row:

            # normalize value to [-255,255]
            pixel = val - lmin
            pixel = pixel / (lmax - lmin)
            pixel = 255 * pixel

            if abs(pixel) > 254:
                char = PIXEL_ASCII_MAP[-1]
            else:
                char = PIXEL_ASCII_MAP[int((abs(pixel) * len(PIXEL_ASCII_MAP) - 1) // 255)]
            if val < 0:
                line += f'{colorama.Fore.RED}{char}{colorama.Style.RESET_ALL}'
            else:
                line += char

        print(line)


def write_img(tensor: mathutil.Tensor, path: str) -> None:
    tensor.normalize()
    with open(path, 'wb') as f:
        w = png.Writer(28, 28, greyscale=True)
        w.write(f, tensor.matrix)
