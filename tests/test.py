import numpy as np
import matplotlib.pyplot as plt
from FPTdiffusion.spt import locate



# FUNCTIONS
def add_heavy_noise(image, noise_level=0.5):
    """
    Adds strong Gaussian noise to a grayscale image.

    Parameters:
    - image: 2D NumPy array (grayscale image)
    - noise_level: standard deviation of noise relative to max(image). Default = 0.5

    Returns:
    - noisy_image: 2D NumPy array with noise added
    """
    image = image.astype(np.float32)
    noise_std = noise_level * np.max(image)
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Keep in displayable range
    return noisy_image

# Main
# make some bullshit image
img = np.zeros((512,512))
nump = 5                            # number of particles.
gen = np.random.default_rng(4)

def Gaussian(x,y,x0=0,y0=0, sigma=0.1):
    return np.exp(- ((x-x0)**2 + (y-y0)**2)/(2*sigma))

xrange = np.linspace(-10, 10,512)
yrange = np.linspace(-10, 10,512)

xv, yv = np.meshgrid(xrange, yrange)

for i in range(nump):
    num1 = gen.uniform(-10, 10)
    num2 = gen.uniform(-10, 10) #np.sqrt(10-num1)
    img+= Gaussian(xv,yv, num1, num2)
    

img = add_heavy_noise(img)

# location
coords = locate(img, 7, 20)

x = coords['x']
y = coords['y']

plt.imshow(img)
plt.scatter(x,y, color='red')
plt.show()


