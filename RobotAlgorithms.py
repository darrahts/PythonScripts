from heapq import *
import numpy


def ConvertGrid(grid):
    '''converts a 2D matrix into a dictionary, where
        keys = x,y coordinate
        values = valid moves to adjacent cells
    '''
    height = len(grid)
    width = len(grid[0])
    graph = dict()
    for j in range(width):
        for i in range(height):
            graph[(i,j)] = []

    for row, col in graph.keys():
        if row < height - 1 and grid[row + 1][col] == 0:
            graph[(row, col)].append(("D", (row + 1, col)))
            graph[(row + 1, col)].append(("U", (row, col)))
        if col < width - 1 and grid[row][col + 1] == 0:
            graph[(row, col)].append(("R", (row, col + 1)))
            graph[(row, col + 1)].append(("L", (row, col)))
    return graph

def Heuristic(cell, goal):
    '''returns the Manhattan distance from the current cell to the goal'''
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def SimpleAStar(graph, start, goal):
    '''AStar algorithm with link costs of 1 and manhattan heuristic'''
    PQ = []
    estDist = Heuristic(start, goal)
    pathCost = 0
    path = ""
    heappush(PQ, (estDist, pathCost, path, start))
    visited = set()
    while len(PQ) > 0:
        estDist, pathCost, path, current = heappop(PQ)
        if current == goal:
            return path
        if current not in visited:
            visited.add(current)
            for dir, neighbor in graph[current]:
                heappush(PQ, (estDist + Heuristic(neighbor, goal), pathCost + 1,
                                    path + dir, neighbor))
    return "No Valid Path found."






if __name__ == "__main__":
    print("yes")
    grid = ([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    print(grid)
    
    graph = ConvertGrid(grid)

    src = (1,1)
    dst = (6,10) #<- path to here
    #dst = (6, 11) #<- no path to here
    
    path = SimpleAStar(graph, src, dst)

    print(path)
















    
