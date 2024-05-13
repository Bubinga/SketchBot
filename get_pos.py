import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush, heappop
from queue import PriorityQueue

def find_connected_components(image):
    # Invert the binary image to find black connected components
    binary_image = cv2.bitwise_not(image)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Remove small clusters
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 8:
            binary_image[labels == label] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    binary_image = cv2.bitwise_not(binary_image)

    return binary_image, num_labels, labels

def find_subarray_from_x_to_end(path, x):
    # Iterate backward through the path array
    for i in range(len(path) - 1, -1, -1):
        if path[i] == x:
            # Return the subarray from index i to the end of the list
            return path[i:]

    # If X is not found, return an empty list
    return []


def dfs_traversal(image, x, y):
    directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

    visited = np.zeros_like(image, dtype=bool)
    visited[y, x] = True

    path = [(x, y)]
    stack = deque([(x, y)])

    while stack:
        current_x, current_y = stack.pop()

        found_next = False
        for dx, dy in directions:
            nx, ny = current_x + dx, current_y + dy

            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] == 0 and not visited[ny, nx]:
                visited[ny, nx] = True 
                path.append((nx, ny))
                stack.append((nx, ny))
                found_next = True

        # If it needs to backtrack, it bfs-s to find the optimal path there
        # Todo update bfs to work with only nodes on path back to next node
        if not found_next and stack:
            next_x, next_y = stack.pop()
            # shortest_path = find_subarray_from_x_to_end(path,(next_x,next_y))
            shortest_path = bfs(image, (current_x, current_y), (next_x, next_y))
            if shortest_path:
                path.extend(shortest_path[1:])  # Exclude the starting point
            visited[next_y,next_x] = True
            stack.append((next_x, next_y))

    return path


def bfs(image, start, end, mask = None):
    directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1,-1)]
    # Initialize visited array and queue for BFS
    visited = np.zeros_like(image, dtype=bool)
    visited[start[1], start[0]] = True
    queue = deque([(start[0], start[1], [])])  # Each element in the queue is (x, y, path)

    while queue:
        x, y, path = queue.popleft()

        # If we reached the end point, return the path
        if (x, y) == end:
            return path + [(x, y)]

        # Explore neighboring pixels
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the neighboring pixel is within image bounds and not visited
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                # if not mask or (nx,ny) in mask:
                visited[ny, nx] = True

                # If the neighboring pixel is black, add it to the queue
                if image[ny, nx] == 0:
                    queue.append((nx, ny, path + [(x, y)]))


def dijkstra(image, start, end):
    directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1,-1)]
    
    # Create a priority queue to store the nodes to be visited
    pq = PriorityQueue()
    pq.put((0, start))  # Priority is the number of white pixels traversed
    visited = set()

    # Initialize distance and path dictionaries
    distance = {start: 0}
    parent = {start: None}

    while not pq.empty():
        # Get the node with the smallest number of white pixels traversed
        curr_cost, curr_node = pq.get()
        visited.add(curr_node)

        # If we reached the end point, reconstruct the path and return it
        if curr_node == end:
            path = []
            while curr_node is not None:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.reverse()
            return path

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = curr_node[0] + dx, curr_node[1] + dy

            # Check if the neighbor is within bounds and not visited
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and (nx, ny) not in visited:
                # Calculate the number of white pixels traversed
                neighbor_cost = curr_cost + (image[ny, nx] == 255)

                # Update distance and parent if this path minimizes white pixels
                if (nx, ny) not in distance or neighbor_cost < distance[(nx, ny)]:
                    distance[(nx, ny)] = neighbor_cost
                    parent[(nx, ny)] = curr_node
                    pq.put((neighbor_cost, (nx, ny)))

    # If no path is found, return an empty list
    return []

def path_generator(image_path, total_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Shrink image
    if image.size > 256**2:
        image = cv2.resize(image, (400, 400))
    _, bw_image = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)

    binary_image, num_labels, labels = find_connected_components(bw_image)
    print("num conn comp:", num_labels)

    current_pos = (0,0)
    #TODO Should update to do nearest cluster instead of random next
    for label in range(1, num_labels):
        # Find starting point for DFS in this cluster
        start_points = np.argwhere(labels == label)
        print("new cluster starting at", start_points[0])

        #traverse min distance to new cluster
        path = dijkstra(binary_image, current_pos, tuple([start_points[0][1],start_points[0][0]]))
        current_pos = path[-1]
        total_path.extend(path)
        print("found path to new cluster", current_pos)

        # Do a DFS traversal within cluster
        path = dfs_traversal(binary_image, current_pos[0], current_pos[1])
        total_path.extend(path)
        current_pos = path[-1]

    print(len(total_path))
    return total_path

def main():
    # display a graph of map
    plt.figure()   
    # plt.imshow(binary_image, cmap='gray')
    # plt.title('Path Traversal')
    plt.axis('off')

    # Load image
    image = cv2.imread('images/dog_256.png', cv2.IMREAD_GRAYSCALE)
    if image.size > 256**2:
        image = cv2.resize(image, (400, 400))
    _, bw_image = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)

    # Convert to pure black and white
    binary_image, num_labels, labels = find_connected_components(bw_image)
    print("num conn comp:", num_labels)

    colors = np.random.randint(0, 256, size=(num_labels, 3), dtype=np.uint8)
    # colors[0] = (72, 136, 240)
    output_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
    # Color each connected component
    for label in range(1, num_labels):
        mask = labels == label
        output_image[mask] = colors[label]

    current_pos = (0,0)
    total_path = [current_pos]
    #TODO Should update to do nearest cluster instead of random next
    for label in range(1, num_labels):
        # Find starting point for DFS in this cluster
        start_points = np.argwhere(labels == label)
        print("starting at", start_points[0])
        #traverse min distance to new cluster
        path = dijkstra(binary_image, current_pos, tuple([start_points[0][1],start_points[0][0]]))
        current_pos = path[-1]
        print("found path to new cluster", current_pos)
        total_path.extend(path[1:]) # slow to slice
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'b-', linewidth=2)

        # Do a DFS traversal within cluster
        path = dfs_traversal(binary_image, current_pos[0], current_pos[1])
        total_path.extend(path[1:]) # slow to slive
        current_pos = path[-1]
        path_x, path_y = zip(*path)
        plt.plot(np.array(path_x), np.array(path_y), 'r-', linewidth=1)
        plt.pause(0.01)

    print("path len",len(total_path))

    plt.imshow(output_image)
    plt.show()


if __name__ == "__main__":
    main()
