import OpenEXR,Imath,vop
import array
import numpy as np
import sys

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(filename):

  file = open(filename, 'r')
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  file.close()
  return np.reshape(data, shape), scale

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

def load_openexr(filename):

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr_img = OpenEXR.InputFile(filename)
    print exr_img.header()
    dw = exr_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = exr_img.channel('R', pt)
    
    red = np.fromstring(redstr, dtype = np.float32)
    red.shape = (size[1], size[0])
    print red.shape, red[10][10]
    save_pfm("%s.pfm" % filename, red)
    return red

def save_openexr(exr_np, filename):

    width, height = exr_np.shape
    out = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))
    out.writePixels({'A': exr_np, 'R' : exr_np, 'G' : exr_np, 'B' : exr_np })

single_channel = load_openexr("./p0.exr")
save_openexr(single_channel, "./save_p0.exr")
single_channel = load_openexr("./save_p0.exr")
