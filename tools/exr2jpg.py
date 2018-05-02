import OpenEXR
import Imath
import Image
import sys

def main(exrfile, jpgfile):
    file = OpenEXR.InputFile(exrfile)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    rgbf = [Image.frombytes("F", size, file.channel(c, pt)) for c in "RGB"]

    extrema = [im.getextrema() for im in rgbf]
    darkest = min([lo for (lo,hi) in extrema])
    lighest = max([hi for (lo,hi) in extrema])
    scale = 255 * 2 / (lighest - darkest)
    def normalize_0_255(v):
        return (v * scale) + darkest
    rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    Image.merge("RGB", rgb8).save(jpgfile)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "usage: exr2jpg <exrfile> <jpgfile>"
    main(sys.argv[1], sys.argv[2])
