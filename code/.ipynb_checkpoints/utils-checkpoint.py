import torch
import numpy as np
import random

def random_policy(td):
    """Helper function to select a random action from available actions"""
#     print("action mask: ", td["action_mask"])
    #print("greedy action: ", torch.argmax(td["action_mask"], 1))
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
#     print("random action: ", action)
    td.set("action", action)
    return td

def greedy(td):
    """Select the action with the highest probability."""
#     print("action mask: ", td["action_mask"])
#     print("end node: ", td["end_node"])
    action = torch.argmax(td["action_mask"], 1)
#     print("greedy action: ", action)
    td.set("action", action)
    return td

def rollout(env, td, policy, max_steps: int = None):
    """Helper function to rollout a policy. Currently, TorchRL does not allow to step
    over envs when done with `env.rollout()`. We need this because for environments that complete at different steps.
    """
    print(td)
    max_steps = float("inf") if max_steps is None else max_steps
    actions = []
    steps = 0
    done_mask = torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

    while not td["done"].all():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]

        # # Select non-done actions
        # td_masked = td.masked_select(~done_mask)

        # # Apply the policy to the non-done actions
        # td_masked = policy(td_masked)

        # # Update the action tensor
        # action = torch.where(~done_mask, td_masked["action"], td["current_node"])

        # # Update td with actions
        # td.set("action", action)
        # actions.append(td["action"])
        
        # print("action: ", action)
        # print("done: ", td["done"])

        # # Perform the next step
        # td = env.step(td)["next"]
        
        # # Update the done mask
        # done_mask |= td["done"]

        steps += 1
        if steps > max_steps:
            print("Max steps reached")
            break
    return (
        env.get_reward(td, torch.stack(actions, dim=1)),
        td,
        torch.stack(actions, dim=1),
    )

def create_connected_graph(num_nodes):
    # Create an adjacency matrix initialized to zero
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Ensure the graph is connected by adding a spanning tree
    for i in range(1, num_nodes):
        # Connect each node to a previous random node to ensure connectivity
        j = random.randint(0, i-1)
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # For undirected graph, make the connection bidirectional

    # Optionally add more edges to make the graph denser
    additional_edges = 20  # You can vary this number to add more or fewer edges
    while additional_edges > 0:
        i = random.randint(0, num_nodes-1)
        j = random.randint(0, num_nodes-1)
        if i != j and adjacency_matrix[i][j] == 0:
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1  # For undirected graph
            additional_edges -= 1

    # import matplotlib.pyplot as plt
    # plt.imshow(adjacency_matrix, cmap='gray', interpolation='none')
    # plt.show()
    
    return adjacency_matrix

def generate_adjacency_matrix(grid_size):
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((grid_size*grid_size, grid_size*grid_size))

    # Iterate over each node
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the index of the current node in the flattened grid
            index = i * grid_size + j

            # Connect the node to its neighbors
            if i > 0:  # Node above
                adjacency_matrix[index, index - grid_size] = 1
            if i < grid_size - 1:  # Node below
                adjacency_matrix[index, index + grid_size] = 1
            if j > 0:  # Node to the left
                adjacency_matrix[index, index - 1] = 1
            if j < grid_size - 1:  # Node to the right
                adjacency_matrix[index, index + 1] = 1

    return adjacency_matrix

def compute_manhattan_distance(grid_size, x, y):
    manhattan_matrix = np.zeros((grid_size * grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            diff_x = abs(i - x)
            diff_y = abs(j - y)
            manhattan_matrix[i*grid_size + j] = 1 / (diff_x + diff_y + 1)
    
    twod_manhattan_matrix = manhattan_matrix.reshape((grid_size, grid_size))

    # import matplotlib.pyplot as plt
    # plt.imshow(twod_manhattan_matrix, cmap='gray', interpolation='none')
    # plt.show()
    # flatten the matrix
    return manhattan_matrix