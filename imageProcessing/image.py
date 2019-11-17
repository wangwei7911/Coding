from PIL import Image
from pylab import *

pil_im = array(Image.open('timg.jfif').convert('L'))

figure()
gray()
contour(pil_im, origin='image')
axis('equal')
axis('off')
figure()
hist(pil_im.flatten(), 128)
show()
