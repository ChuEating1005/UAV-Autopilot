import numpy as np
import cv2
import matplotlib.pyplot as plt

### Part 1 ### 
def bluepass_filter(img):
    newimg = np.zeros(img.shape, img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B, G, R = img[i,j,0], img[i,j,1], img[i,j,2]

            if B > 100 and 0.6 * B > G and 0.6 * B > R:
                newimg[i,j] = img[i,j]
            else:
                newimg[i,j] = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    return newimg

def contrast_brightness(img, contrast=100, brightness=40):
    newimg = np.zeros(img.shape, dtype=np.int32)
    img = img.astype(np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B, G, R = img[i,j,0], img[i,j,1], img[i,j,2]
            if (B > 100 and 0.6 * B > G and 0.6 * B > R) or ((B + G) * 0.3 > R):
                newimg[i, j] = ((img[i, j] - 127) * (contrast / 127 + 1) + 127 + brightness)
            else:
                newimg[i, j] = img[i, j]
    newimg = np.clip(newimg, 0, 255).astype(np.uint8)
    return newimg    

def part1():
    img = cv2.imread('./data/test.jpg')

    newimg1 = bluepass_filter(img)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Part 1-1', fontsize=17)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(newimg1, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Bluepass Filter')
    ax[1].axis('off')
    fig.tight_layout()
    plt.savefig('./output/1-1_compare.png')
    plt.show()
    cv2.imwrite('./output/1-1.jpg', newimg1)

    newimg2 = contrast_brightness(img)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Part 1-2', fontsize=17)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(newimg2, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Contrast and Brightness')
    ax[1].axis('off')
    fig.tight_layout()
    plt.savefig('./output/1-2_compare.png')
    plt.show()
    cv2.imwrite('./output/1-2.jpg', newimg2)


### Part2 ###
def bilinear_interpolation(img, scale=3):
    newimg = np.zeros((img.shape[0] * scale, img.shape[1] * scale, 3), img.dtype)
    for i in range(newimg.shape[0]):
        for j in range(newimg.shape[1]):
            x, y = i / scale, j / scale
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, img.shape[0] - 1), min(y1 + 1, img.shape[1] - 1)
            a, b = x - x1, y - y1
            newimg[i, j] = (1 - a) * (1 - b) * img[x1, y1] + a * (1 - b) * img[x2, y1] + (1 - a) * b * img[x1, y2] + a * b * img[x2, y2]
    return newimg
def part2():
    img = cv2.imread('./data/test.jpg')
    # print(img.shape)
    newimg = bilinear_interpolation(img)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Part 2', fontsize=17)
    ax[0].imshow(cv2.cvtColor(img[200:300, 250:350, :], cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(newimg[600:900, 750:1050, :], cv2.COLOR_BGR2RGB))
    ax[1].set_title('Bilinear Interpolation')
    ax[1].axis('off')
    fig.tight_layout()
    plt.savefig('./output/2_compare.png')
    plt.show()
    cv2.imwrite('./output/2.jpg', newimg)

### Part3 ###
def sobel_filter(img):
    # Create gradient kernels
    gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gy_kernel = gx_kernel.T

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    img_gx = cv2.filter2D(img_blur, -1, gx_kernel).astype(np.int32)
    img_gy = cv2.filter2D(img_blur, -1, gy_kernel).astype(np.int32)
    img_edge = np.sqrt(img_gx ** 2 + img_gy ** 2).astype(np.uint8)
    img_edge = np.clip(img_edge, 0, 255)

    cv2.imwrite('./output/3_gx.jpg', img_gx)
    cv2.imwrite('./output/3_gy.jpg', img_gy)
    cv2.imwrite('./output/3_edge.jpg', img_edge)
    return img_edge

def part3():
    img = cv2.imread('./data/test.jpg')
    edge = sobel_filter(img)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Part 3', fontsize=17)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(edge, cmap='gray')
    ax[1].set_title('Edge Detection')
    ax[1].axis('off')
    fig.tight_layout()
    plt.savefig('./output/3_compare.png')
    plt.show()

if __name__ == '__main__':
    part1()
    part2() 
    part3()