import numpy as np

# Input image
image = np.array([[3, 1, 2, 0, 1],
                  [0, 1, 3, 2, 0],
                  [1, 2, 1, 0, 2],
                  [2, 1, 0, 1, 3],
                  [1, 0, 1, 2, 1]])

# Kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

def convolve(image, kernel):
h, w = image.shape
kh, kw = kernel.shape

pad = kh // 2
padded_image = np.pad(image, pad_width= pad, mode="constant", constant_values = 0)
output = np.zeros_like(image)

for i in range(h):
    for j in range(w):
        region = padded_image[i:i+kh, j:j+kw]
        output[i, j] = np.sum(region * kernel)

    return output


# output = np.zeros_like(image)

# for i in range(h):
# for j in range(w):
# region = image[i:i+kh, j:j+kw]
# output[i, j] = np.sum(region * kernel)

# return output


# Perform convolution
output = convolve(image, kernel)
print(output)