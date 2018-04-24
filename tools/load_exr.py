import OpenEXR,Imath,vop
import array
import numpy as np
import sys

def save_pfm(filename, image, scale = 1):
  file = open(filename, 'w')
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file) 
  file.close()


pt = Imath.PixelType(Imath.PixelType.FLOAT)
exr_img = OpenEXR.InputFile("R.0040.exr")
print exr_img.header()
dw = exr_img.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
redstr = exr_img.channel('R', pt)

red = np.fromstring(redstr, dtype = np.float32)
red.shape = (size[1], size[0])
print red.shape, red[10][10]
save_pfm("test.pfm", red)
