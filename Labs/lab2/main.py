import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

random.seed(0)

def histogram(image):
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[int(image[i,j])] += 1
    return hist

def transformation(image, h, n):
    t = np.zeros(256)
    for i in range(256):
        for j in range(i):
            t[i] += h[j]
        t[i] = 255 * t[i] / n
    return t[image]

def histogram_equalization(image, color_space):
    if color_space == 'BGR':
        B, G, R = cv2.split(image)

        B_hist = histogram(B.astype(np.float32))
        G_hist = histogram(G.astype(np.float32))
        R_hist = histogram(R.astype(np.float32))

        B_eq = transformation(B, B_hist, B.shape[0] * B.shape[1]).astype(np.uint8)
        G_eq = transformation(G, G_hist, G.shape[0] * G.shape[1]).astype(np.uint8)
        R_eq = transformation(R, R_hist, R.shape[0] * R.shape[1]).astype(np.uint8)

        return cv2.merge((B_eq, G_eq, R_eq))
    else: # color_space == 'HSV'
        H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        V_hist = histogram(V.astype(np.float32))
        V_eq = transformation(V, V_hist, V.shape[0] * V.shape[1]).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((H, S, V_eq)), cv2.COLOR_HSV2BGR)

### Part 1: Historgram Equalization ###
def part1():
    original = cv2.imread('./data/histogram.jpg')
    # 1-a: Apply histogram equalization to BGR channels separately
    equalized_a = histogram_equalization(original, 'BGR')
    # 1-b: Apply histogram equalization to V channel in HSV color space
    equalized_b = histogram_equalization(original, 'HSV')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparison', fontsize=16)
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(equalized_a, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Equalized (BGR)')
    ax[1].axis('off')
    ax[2].imshow(cv2.cvtColor(equalized_b, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Equalized (HSV)')
    ax[2].axis('off')
    fig.tight_layout()
    plt.savefig('./output/part1.png')
    plt.show()

    cv2.imwrite('./output/part1_original.jpg', original)
    cv2.imwrite('./output/part1_equalized_BGR.jpg', equalized_a)
    cv2.imwrite('./output/part1_equalized_HSV.jpg', equalized_b)
    
### Part 2: Otsu Thresholding ###
def otsu_thresholding(img, hist):
    min_sigma = float('inf')
    threshold = 0

    for i in range(1, 255):
        black = np.array([])
        white = np.array([])
        for j in range(i):
            black = np.append(black, np.full(int(hist[j]), j))
        for j in range(i, 256):
            white = np.append(white, np.full(int(hist[j]), j))
        
        varB = np.var(black)
        varW = np.var(white)

        if len(black) == 0 or len(white) == 0:
            continue

        temp_sigma = varB + varW
        # print(temp_sigma)
        if temp_sigma < min_sigma:
            min_sigma = temp_sigma
            threshold = i
            
    
    return np.where(img < threshold, 0, 255)


def part2():
    original = cv2.imread('./data/input.jpg')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hist = histogram(original)
    binarized = otsu_thresholding(original, hist)
    cv2.imwrite('./output/part2.jpg', binarized.astype(np.uint8))

### Part3: Connected Component Algorithm: 2-pass ###
def random_rgb():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def find_root(labels, i):
    while labels[i] != i:
        i = labels[i]
    return i

def union(label_equivalences, i, j):
    root_i = find_root(label_equivalences, i)
    root_j = find_root(label_equivalences, j)
    if root_i != root_j:
        label_equivalences[root_j] = root_i

def two_pass(img):
    labels = np.zeros_like(img, dtype=int)
    
    next_label = 1
    label_equivalences = {}
    
    # First pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                continue

            neighbors = []
            if i > 0 and img[i-1, j] != 0:
                neighbors.append(labels[i-1, j])
            if j > 0 and img[i, j-1] != 0:
                neighbors.append(labels[i, j-1])

            if len(neighbors) == 0:
                labels[i, j] = next_label
                label_equivalences[next_label] = next_label
                next_label += 1
            elif len(neighbors) == 1:
                labels[i, j] = neighbors[0]
            else:
                labels[i, j] = min(neighbors)
                union(label_equivalences, neighbors[0], neighbors[1])
    
    # Second pass
    colored_labels = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color_map = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                colored_labels[i, j] = [0, 0, 0]
            else:
                label = find_root(label_equivalences, labels[i, j])
                if label not in color_map:
                    color_map[label] = random_rgb()
                colored_labels[i, j] = color_map[label]
        
    return colored_labels

def part3():
    original = cv2.imread('./data/input.jpg')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    binarized = otsu_thresholding(original, histogram(original)).astype(np.uint8)
    connected_components = two_pass(binarized)
    cv2.imwrite('./output/part3.jpg', connected_components.astype(np.uint8))

if __name__ == '__main__':
    part1()
    part2()
    part3()
