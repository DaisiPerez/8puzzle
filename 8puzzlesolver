import heapq
import itertools
from collections import deque
import pandas as pd
import pprint


"""1. Can we transfer this problem into a general graph search problem? (node & edge)
The 8-puzzle problem can very well be represented as a graph search problem.
Each node represents a state of the 3×3 board. An edge exists between two nodes if one state can be transitioned by sliding a tile into an empty space.
There are 362,880 possible board configurations. The problem is only solvable if the initial state is reachable through valid moves.
Each move should have an equal and uniform cost, meaning this should be an unweighted graph!
Since the problem is naturally a graph traversal/search problem, we must formulate solutions using graph search algorithms,
such as greedy best-first search, dijkstra's algorithm search, and A* search.
"""
class Puzzle:
   def __init__(self, start, goal):
       self.start = tuple(start)
       self.goal = tuple(goal)
       self.n = 3  # 3x3 grid


   def get_neighbors(self, state):
       neighbors = []
       empty_idx = state.index(0)  # Find the empty tile
       row, col = divmod(empty_idx, self.n)
      
       # Possible moves: Up, Down, Left, Right
       moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
      
       for dr, dc in moves:
           new_row, new_col = row + dr, col + dc
           if 0 <= new_row < self.n and 0 <= new_col < self.n:
               new_idx = new_row * self.n + new_col
               new_state = list(state)
               new_state[empty_idx], new_state[new_idx] = new_state[new_idx], new_state[empty_idx]
               neighbors.append(tuple(new_state))
      
       return neighbors


   def bfs(self):
       """BFS"""
       queue = deque([(self.start, [self.start])])
       visited = set()


       while queue:
           state, path = queue.popleft()
           if state == self.goal:
               return path  # Return moves to reach goal
           if state in visited:
               continue
           visited.add(state)
           for neighbor in self.get_neighbors(state):
               queue.append((neighbor, path + [neighbor]))


       return None  # No solution found


   def dfs(self):
       """DFS"""
       stack = [(self.start, [self.start])]
       visited = set()


       while stack:
           state, path = stack.pop()
           if state == self.goal:
               return path
           if state in visited:
               continue
           visited.add(state)
           for neighbor in self.get_neighbors(state):
               stack.append((neighbor, path + [neighbor]))


       return None  # No solution found
  
   """2. Solve this problem using dynamic construction of the search space.
BFS found the best solution in 4 moves, while DFS explored a much longer path, reaching the solution after 100+ times!
DFS is much faster initially but is not efficient in structured problems like the 8-puzzle, where optimality and conciseness is desired.
It is best to use BFS when you need the shortest path. Use DFS if memory is a concern since it uses less memory than BFS.
Additionally, DFS is best used when the solution is very deep in the tree.
"""


   def heuristic(self, state):
       """Manhattan Distance heuristic"""
       distance = 0
       for i in range(1, self.n * self.n):
           x1, y1 = divmod(state.index(i), self.n)
           x2, y2 = divmod(self.goal.index(i), self.n)
           distance += abs(x1 - x2) + abs(y1 - y2)
       return distance




   def greedy_best_first(self):
       """Greedy Best-First Search"""
       pq = [(self.heuristic(self.start), self.start, [self.start])]
       visited = set()


       while pq:
           _, state, path = heapq.heappop(pq)
           if state == self.goal:
               return path
           if state in visited:
               continue
           visited.add(state)
           for neighbor in self.get_neighbors(state):
               heapq.heappush(pq, (self.heuristic(neighbor), neighbor, path + [neighbor]))


       return None
   """3. How to Build a Weighted Tree for the 8-Puzzle Problem?
   The root of the tree is the starting puzzle configuration.
   Each move (up, down, left, right) creates a new child node. Each move has a cost of 1.
   The weight is the sum of actual cost, which is the amount of moves taken and estimated cost to the best solution.
   The weight is based on heuristic evaluation alone.
   The tree expands as nodes are visited, adding paths toward the best solution.
"""


   """Greedy Best-First Search found the best solution in only 4 moves. It prioritizes expanding nodes with the lowest heuristic cost.
   However, It does not guarantee an optimal solution but is the fastest compared to the other algorithms."""


   def dijkstra(self):
       """Dijkstra"""
       pq = [(0, self.start, [self.start])]
       visited = set()


       while pq:
           cost, state, path = heapq.heappop(pq)
           if state == self.goal:
               return path
           if state in visited:
               continue
           visited.add(state)
           for neighbor in self.get_neighbors(state):
               heapq.heappush(pq, (cost + 1, neighbor, path + [neighbor]))


       return None
   """Dijkstra's algorithm found the best solution in 4 moves, same as the Greedy Best-First Search. It treats each move equally with the same cost.
     For this reason, it finds the shortest path to the best solution.
     It explores more nodes than the Greedy Best-First Search and as a result, it makes it slightly slower."""


   def a_star(self):
       """A*"""
       pq = [(self.heuristic(self.start), 0, self.start, [self.start])]
       visited = set()


       while pq:
           _, cost, state, path = heapq.heappop(pq)
           if state == self.goal:
               return path
           if state in visited:
               continue
           visited.add(state)
           for neighbor in self.get_neighbors(state):
               new_cost = cost + 1
               heapq.heappush(pq, (new_cost + self.heuristic(neighbor), new_cost, neighbor, path + [neighbor]))


       return None
   """A* Search found the best solution in 4 moves, same as the Greedy Best-First Search and Dijkstra's algorithm.
   It combines the strengths of both the euristic-based exploration of Greedy Best-First Search and optimibablity of Dijkstra’s Algorithm.
   Because of this, It's quicker, optimal, and efficient!
   A* Search finds the shortest path without unnecessary exploration of nodes."""


# Verify the correctness of each search algorithm


# Define test cases
test_start_state = (1, 2, 3, 4, 6, 8, 7, 5, 0)  # Given initial state
test_goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # Goal configuration


# Create a Puzzle instance
test_puzzle = Puzzle(test_start_state, test_goal_state)


# Run and validate each search algorithm
test_results = {
   "BFS": test_puzzle.bfs(),
   "DFS": test_puzzle.dfs(),
   "Greedy Best-First": test_puzzle.greedy_best_first(),
   "Dijkstra": test_puzzle.dijkstra(),
   "A*": test_puzzle.a_star(),
}


# Format and print results in dictionary format
formatted_results = {k: v if v else "No Solution" for k, v in test_results.items()}
pprint.pprint(formatted_results)
