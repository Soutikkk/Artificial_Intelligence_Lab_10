
import heapq

# Grid: 0 = walkable, 1 = obstacle
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

# Manhattan distance heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])

    # Priority queue: (f_score, position)
    open_list = [(heuristic(start, goal), start)]

    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        r, c = current

        # Check four possible directions
        neighbors = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        ]

        for nr, nc in neighbors:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if grid[nr][nc] == 1:
                continue

            neighbor = (nr, nc)
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    return None

# Find the shortest path
path = astar(grid, start, goal)

if path:
    print("Path found!")
    print("Path:", path)
    print("Path cost:", len(path) - 1)

    # Display the path on the grid
    for r, c in path:
        if (r, c) != start and (r, c) != goal:
            grid[r][c] = 2

    print("\nGrid representation:")
    for r, row in enumerate(grid):
        line = []
        for c, cell in enumerate(row):
            if (r, c) == start:
                line.append("S")
            elif (r, c) == goal:
                line.append("G")
            elif cell == 1:
                line.append("#")
            elif cell == 2:
                line.append("*")
            else:
                line.append(".")
        print(" ".join(line))
else:
    print("No path found!")
