import collections
import heapq

class Graph:
    def __init__(self):
        self.adj = collections.defaultdict(list)

    def add_pipe(self, u, v, cost):
        self.adj[u].append((v, cost))
        self.adj[v].append((u, cost))

def uniform_cost_search(graph, start, end):
    pq = [(0, start, [start])]
    
    visited = set()
    
    while pq:
        cost, current_node, path = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == end:
            return path, cost, len(visited)
            
        for neighbor, pipe_cost in graph.adj.get(current_node, []):
            if neighbor not in visited:
                new_cost = cost + pipe_cost
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, neighbor, new_path))
                
    return None, 0, len(visited)

def main():
    pipe_network = Graph()
    
    pipes_data = [
        (0, 1, 5), (0, 2, 8), (1, 3, 6), (2, 3, 7),
        (2, 4, 3), (3, 5, 4), (4, 5, 9), (4, 6, 5),
        (5, 6, 11)
    ]
    for u, v, cost in pipes_data:
        pipe_network.add_pipe(u, v, cost)

    start_junction = 4
    cheese_junction = 0
    
    print(f"Help Terry find the cheapest cost from {start_junction} to {cheese_junction}!")
    
    path, cost, visited_count = uniform_cost_search(pipe_network, start_junction, cheese_junction)
    
    print("\n--- Uniform Cost Search ---")
    if path:
        print(f" Path: {' -> '.join(map(str, path))}")
        print(f" Total Cost: {cost}")
        print(f" Junctions Visited: {visited_count}\n")
    else:
        print("No path found!\n")

if __name__ == "__main__":
    main()