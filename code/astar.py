from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
import torch

class AStarSearch:
    def __init__(self, env):
        self.env = env
        self.observation = self.env._reset(batch_size=1)
        self.start = self.observation["first_node"]
        self.end = self.observation["end_node"]
        self.locations = self.observation["locs"]
        self.edges = self.observation["edges"]
    
    @staticmethod
    def heuristic(a, b):
        return torch.sum(torch.abs(a - b), dim=-1)
    
    def search(self):
        frontier = PriorityQueue()
        start_coord = self.locations[0, self.start]
        end_coord = self.locations[0, self.end]
        frontier.put(self.start.item(), 0)
        came_from = {self.start.item(): None}
        cost_so_far = {self.start.item(): 0}

        while not frontier.empty():
            current = frontier.get()
            if current == self.end.item():
                break

            neighbors = np.argwhere(self.edges[0, current])
            for neighbor in neighbors[0]:
                neighbor = neighbor.item()
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(self.locations[0, current], end_coord)
                    frontier.put(neighbor, priority)
                    came_from[neighbor] = current
        
        return came_from, cost_so_far

    def render(self, came_from):
        _, ax = plt.subplots()
        path = []
        current = self.end.item()
        while current != self.start.item():
            path.append(current)
            current = came_from[current]
        path.append(self.start.item())
        path.reverse()
        
        actions = torch.tensor(path)
        a_locs = [self.locations[0, p].numpy() for p in path]
        a_locs = np.array(a_locs)
        x, y = a_locs[:, 0], a_locs[:, 1]
        
        locations_numpy = self.locations.squeeze().numpy()
        x_i, y_i = locations_numpy[:, 0], locations_numpy[:, 1]
        ax.scatter(x_i, y_i, color="tab:blue")
        
        edges_numpy = self.edges.squeeze().numpy()
        for i in range(edges_numpy.shape[0]):
            for j in range(edges_numpy.shape[1]):
                if edges_numpy[i, j]:
                    ax.plot([x_i[i], x_i[j]], [y_i[i], y_i[j]], color='g', alpha=0.1)
        
        dx, dy = np.diff(x), np.diff(y)
        ax.quiver(x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="r", alpha=1.0)

        ax.scatter(x[-1], y[-1], color="tab:red", s=100, edgecolors="black", zorder=10)
        
        ax.scatter(x[0], y[0], color="tab:green", s=100, edgecolors="black", zorder=10)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
