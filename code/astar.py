from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
import torch

class AStarSearch:
    def __init__(self, td):
        self.start = td["first_node"]
        self.end = td["end_node"]
        self.locations = td["locs"]
        self.edges = td["edges"]
        self.batch_size = self.start.shape[0]
    
    @staticmethod
    def heuristic(a, b):
        return torch.sum(torch.abs(a - b), dim=-1)
    
    def search(self):
        batch_size = self.start.shape[0]
        came_froms = [{} for _ in range(batch_size)]
        cost_so_fars = [{} for _ in range(batch_size)]
        paths = [None for _ in range(batch_size)]
        rewards = torch.zeros(batch_size)

        max_path_length = 0
        for i in range(batch_size):
            frontier = PriorityQueue()
            start = self.start[i].item()
            end = self.end[i].item()
            start_coord = self.locations[i, start]
            end_coord = self.locations[i, end]
            frontier.put(start, 0)
            came_froms[i][start] = None
            cost_so_fars[i][start] = 0

            while not frontier.empty():
                current = frontier.get()
                if current == end:
                    break

                neighbors = (self.edges[i, current] > 0).nonzero(as_tuple=True)[0]
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    new_cost = cost_so_fars[i][current] + 1
                    if neighbor not in cost_so_fars[i] or new_cost < cost_so_fars[i][neighbor]:
                        cost_so_fars[i][neighbor] = new_cost
                        priority = new_cost + self.heuristic(self.locations[i, current], end_coord)
                        frontier.put(neighbor, priority)
                        came_froms[i][neighbor] = current

            paths[i] = self.reconstruct_path(came_froms[i], start, end)
            max_path_length = max(max_path_length, len(paths[i]))

            # Calculate the total reward for each path
            rewards[i] = -cost_so_fars[i][end]

        # Pad the paths to the maximum length in the batch
        for i in range(batch_size):
            paths[i] += [paths[i][-1]] * (max_path_length - len(paths[i]))

        results = {
            "actions": torch.tensor(paths),
            "reward": rewards
        }
        return results

    def reconstruct_path(self, came_from, start, end):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

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
