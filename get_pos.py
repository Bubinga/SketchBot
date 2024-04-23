import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def find_connected_components(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Invert the binary image to find black connected components
    binary_image = cv2.bitwise_not(binary_image)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    print(f"{num_labels=}")
    # Remove small clusters
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 8:
            print("changing")
            print(binary_image[labels==label])
            binary_image[labels == label] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    print(f"{num_labels=}")
    binary_image = cv2.bitwise_not(binary_image)
    return binary_image

def dfs_traversal(image, x, y):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    visited = np.zeros_like(image, dtype=bool)
    visited[y, x] = True

    path = [(x, y)]
    stack = deque([(x, y)])

    while stack:
        current_x, current_y = stack.pop()

        for dx, dy in directions:
            nx, ny = current_x + dx, current_y + dy

            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] == 0 and not visited[ny, nx]:
                visited[ny, nx] = True
                path.append((nx, ny))
                stack.append((nx, ny))

    return path

def main():
    # Load image
    image = cv2.imread('dog_256_256.png', cv2.IMREAD_GRAYSCALE)
    _, bw_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Convert to pure black and white
    binary_image = find_connected_components(bw_image)

    counter = 0
    for x in range(256):
        for y in range(256):
            if binary_image[x][y] != bw_image[x][y]:
                counter += 1
    print(f"{counter=}")
    # Create a canvas for plotting
    plt.figure()
    plt.imshow(binary_image, cmap='gray')
    plt.title('DFS Traversal')
    plt.axis('off')
    plt.show()

    # should update to do traversal on each cluster
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 0:
                path = dfs_traversal(binary_image, x, y)
                path_x, path_y = zip(*path)
                print(path_x,path_y)
                plt.plot(path_x, path_y, 'r-', linewidth=1)
                plt.pause(0.0001)

    plt.show()

if __name__ == "__main__":
    main()
