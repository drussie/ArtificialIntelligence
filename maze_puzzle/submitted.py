# submitted.py
# ---------------
# Licensing Information:
# This HW is inspired by previous work by University of Illinois at Urbana-Champaign


"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from queue import Queue

def bfs(maze):
    # https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/?ref=gcse
    from collections import deque

    start = maze.start
    waypoint = maze.waypoints[0]
    queue = deque([(start, [start])])   # Create a queue for BFS - stores the current position and the path to it
    visited = set() # set to store the visited positions

    while queue:
        (current_position, path) = queue.popleft() # get the current position and the path to it
        if current_position in visited:
            continue

        visited.add(current_position)   # Mark the current position as visited

        if current_position == waypoint:    # are we at the goal?
            return path # return the path

        # Add the neighbors of the current position to the queue
        neighbors = maze.neighbors(*current_position)
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor])) # add the neighbor to the queue with the path to it

    return None

# https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/a-star-search-algorithm/
def astar_single(maze):
    import heapq
    from math import sqrt

    def heuristic(a, b):
        # Using Euclidean distance between points a and b
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    start = maze.start  # Starting position
    waypoint = maze.waypoints[0]  # Waypoint position
    open_list = set([start])
    closed_list = set([])
    g = {start: 0}
    parents = {start: start}

    while open_list:
        n = None

        # Find a node with the lowest value of f() - evaluation function
        for v in open_list:
            if n is None or g[v] + heuristic(v, waypoint) < g[n] + heuristic(n, waypoint):
                n = v

        if n is None:
            return None

        # If the current node is the waypoint
        # then we begin reconstructing the path from it to the start node
        if n == waypoint:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            return reconst_path

        # For all neighbors of the current node do
        for neighbor in maze.neighbors(*n):
            if neighbor not in open_list and neighbor not in closed_list:
                open_list.add(neighbor)
                parents[neighbor] = n
                g[neighbor] = g[n] + 1  # Assuming each step costs 1
            else:
                if g[neighbor] > g[n] + 1:
                    g[neighbor] = g[n] + 1
                    parents[neighbor] = n

                    if neighbor in closed_list:
                        closed_list.remove(neighbor)
                        open_list.add(neighbor)

        # Remove n from the open_list, and add it to closed_list
        open_list.remove(n)
        closed_list.add(n)

    return None

def dfs(maze):
    start = maze.start  # Starting position
    waypoint = maze.waypoints[0]  # Waypoint position
    stack = [(start, [start])]  # Stack for DFS as per PPT presentation, storing (current_position, path_to_current_position)
    visited = set()  # Set to store visited positions
    
    while stack:
        (current_position, path) = stack.pop()  # Get the current position and the path to it
        if current_position in visited:
            continue
        
        visited.add(current_position)  # Mark the current position as visited

        if current_position == waypoint:  # Are we at the gtoal
            return path  # Return the path

        # Add the neighbors of the current position to the queue
        neighbors = maze.neighbors(*current_position)
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))  # Add neighbor to stack with the path to it

    return None  # If no path is found, return None

