import heapq
import math

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.h = 0 
        self.f = 0 

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def get_neighbors(grid, node, allow_diagonals=False):
    neighbors = []
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
    ]
    if allow_diagonals:
        directions.extend([
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ])

    for move in directions:
        node_pos = (node.position[0] + move[0], node.position[1] + move[1])

        if not (0 <= node_pos[0] < len(grid) and 0 <= node_pos[1] < len(grid[0])):
            continue
        
        if grid[node_pos[0]][node_pos[1]] == '1':
            continue
        
        neighbors.append(Node(node, node_pos))
    return neighbors

def get_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]

def greedy_best_first_search(grid, start, end, heuristic_name='manhattan', allow_diagonals=False, ghost_zones=None):
    if ghost_zones is None:
        ghost_zones = {}

    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_set = set()
    nodes_explored = 0

    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)
        nodes_explored += 1

        if current_node == end_node:
            path = get_path(current_node)
            return path, len(path) - 1, nodes_explored

        closed_set.add(current_node.position)

        for neighbor in get_neighbors(grid, current_node, allow_diagonals):
            if neighbor.position in closed_set:
                continue

            dx = abs(neighbor.position[0] - end_node.position[0])
            dy = abs(neighbor.position[1] - end_node.position[1])
            
            if heuristic_name == 'euclidean':
                neighbor.h = math.sqrt(dx**2 + dy**2)
            elif heuristic_name == 'diagonal':
                neighbor.h = dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
            else:
                neighbor.h = dx + dy
            
            neighbor.f = neighbor.h

            if not any(open_node for _, open_node in open_list if open_node == neighbor):
                heapq.heappush(open_list, (neighbor.f, neighbor))
                
    return None, 0, nodes_explored 

if __name__ == "__main__":
    # S: Start, G: Goal, 1: Wall, 0: Open, Z: Ghost Zone
    haunted_house_grid = [
        ['S', '0', '0', '1', '0', '0'],
        ['1', '1', '0', '1', 'G', '0'],
        ['0', '0', '0', '1', '0', '0'],
        ['0', '1', '1', 'Z', '1', '1'],
        ['0', '0', '0', 'Z', '0', '0']
    ]
    start_pos = (0, 0)
    end_pos = (1,2) 
    ghost_zones_config = {(3, 3): 5, (4, 3): 5}

    print("--- Pathfinding with Greedy Best-First Search ---")
    
    path_g, len_g, exp_g = greedy_best_first_search(haunted_house_grid, start_pos, end_pos)

    print(f"\nGreedy Search Results:")
    if path_g:
        ghosts_spotted_at = []
        for pos in path_g:
            if haunted_house_grid[pos[0]][pos[1]] == 'Z':
                ghosts_spotted_at.append(pos)
        
        if ghosts_spotted_at:
            print(f"Ghost spotted at: {ghosts_spotted_at}")
        else:
            print("No ghosts were spotted on the path.")

        print(f"Path Found: {path_g}")
        print(f"Path Length: {len_g}")
        print(f"Nodes Explored: {exp_g}")
    else:
        print("No path found.")
        print(f"Nodes Explored: {exp_g}")