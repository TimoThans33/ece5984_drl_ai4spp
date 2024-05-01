from typing import Optional
import torch
import torch.nn as nn
import random

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo import AttentionModel, AutoregressivePolicy
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

from utils import compute_manhattan_distance, generate_adjacency_matrix

def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
    """Reset the environment to the initial state"""
    # If no TensorDict is provided, generate a new one
    init_locs = td["locs"] if td is not None else None
    init_edges = td["edges"] if td is not None else None
    # If no batch_size is provided, use the batch_size of the initial locations
    if batch_size is None:
        batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
    # If no device is provided, use the device of the initial locations
    device = init_locs.device if init_locs is not None else self.device 
    self.to(device)
    # If no initial locations are provided, generate new ones
    if init_locs is None:
        grid_out = self.generate_data(batch_size=batch_size).to(device)
        init_locs = grid_out["locs"]
        init_edges = grid_out["edges"]

    # Reset step count
    step_count = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # If batch_size is an integer, convert it to a list
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

    # Get the number of locations
    num_loc = init_locs.shape[-2]
    # print("num_loc: ", num_loc)

    # Initialize a start and end node
    first_node = torch.randint(0, num_loc, (batch_size), device=device)
    # Initialize the end node to a random node until it is unequal to the start node
    while True:
        end_node = torch.randint(0, num_loc, (batch_size), device=device)
        if not torch.any(torch.eq(first_node, end_node)):
            break


    # print("end_node: ", end_node)
    # print("first_node: ", first_node)
    
    batch_indices = torch.arange(len(first_node))
    available = init_edges[batch_indices, first_node]

    # Compute the distance to the target
    target = init_locs[batch_indices, end_node].unsqueeze(1)
    distance_to_target = torch.norm(init_locs - target, dim=-1, keepdim=True)

    # Initialize the index of the current node 
    i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

    # Initialize the done mask
    done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    return TensorDict(
        {
            "locs": init_locs,
            "edges": init_edges,
            "manhattan_matrix": distance_to_target,
            "first_node": first_node,
            "current_node": first_node,
            "end_node": end_node,
            "step_count": step_count,
            "i": i,
            "action_mask": available,
            "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            "done_mask": done_mask,
        },
        batch_size=batch_size,
    )

def _step(self, td: TensorDict) -> TensorDict:
    # Only update the non-done elements
    done_mask = td["done_mask"]
    #td_masked = td.masked_select(~done_mask)
    #print("td_masked action: ", td_masked["action"])
    #print("td current node: ", td["current_node"])
    current_node = torch.where(~done_mask, td["action"], td["current_node"])
    # print("current_node: ", current_node)
    # update the step count only on non-done elements
    step_count = td["step_count"]
    step_count = torch.where(~done_mask, step_count + 1, step_count)
    # print("step_count: ", step_count)

    # current_node = td["action"]


    # output visiting node
    # first_node = current_node if td["i"].all() == 0 else td["first_node"]

    # Mark the current node as visited
    available = get_action_mask(self, td)

    # Mask the current node to prevent revisiting
    available = available & ~td["current_node"].unsqueeze(-1).eq(torch.arange(available.shape[-1], device=available.device))
#     print("Available pre invalidation:", available)
    
    # these are all valid neighbors
    valid_indices = (available == True).nonzero()

    # Group the indices by batch
    batch_size = available.size(0)
    valid_indices_list = []
    for i in range(batch_size):
        batch_indices = valid_indices[valid_indices[:, 0] == i][:, 1]
        valid_indices_list.append(batch_indices)

#     # Convert the list of tensors to a single tensor with proper format
    valid_indices_list = [indices.tolist() for indices in valid_indices_list]
#     print("valid indices tensor", valid_indices_list)
    
    invalidation_prob = 0.5
    
    available_float = available.float()
    
    invalidate_mask = torch.rand_like(available_float) < invalidation_prob
    # Apply the invalidation mask to invalidate random neighbor nodes
    available[invalidate_mask] = False
    
    # Loop over each batch
    for batch_index in range(len(valid_indices_list)):
        # Check if any actions are available for the current batch
        if not available[batch_index].any():
#             print(f"No actions available for batch {batch_index}")

            # Choose a random valid index to re-enable for this batch
            valid_indices = valid_indices_list[batch_index]
#             print("valid indices", valid_indices)
            reenable_index = random.choice(valid_indices)
#             print("reenable_index", reenable_index)

            # Re-enable the selected valid action
            available[batch_index, reenable_index] = True
    
    
#     print("Available post invalidation", available)

  
    # Create a tensor of batch indices
    # batch_indices = torch.arange(len(current_node))

    # Use advanced indexing to get the neighbors for each batch
    # neighbors = td["edges"][batch_indices, current_node]

    # Update the action_mask to only include the current node's neighbors
    # available = available & neighbors
 
    done = current_node == td["end_node"]
    done |= step_count >= 100
    # print("done: ", done)

    # done = torch.sum(td["action_mask"], dim=-1) == 0
    
    done_mask = done_mask | done
    # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
    reward = torch.zeros_like(done)

    td.update(
        {
            # "first_node": first_node,
            "step_count": step_count,
            "current_node": current_node,
            "i": td["i"] + 1,
            "action_mask": available,
            "reward": reward,
            "done": done,
            "done_mask": done_mask, 
        },
    )
    return td

def get_action_mask(self, td: TensorDict) -> TensorDict:
    # Get the current node
    current_node = td["action"]

    # Create a tensor of batch indices
    batch_indices = torch.arange(len(current_node))

    # Use advanced indexing to get the neighbors for each batch
    neighbors = td["edges"][batch_indices, current_node]

    # Element-wise multiplication with the manhattan distance
    # available = neighbors * td["manhattan_distance"][batch_indices, current_node]
    available = neighbors

    # print("available: ", available)

    # Apply the heuristic to the neighbors to get the action mask
    # get the locations of the neighbors
    # locs = td["locs"]
    # locs_ordered = gather_by_index(locs, neighbors)
    return available

def get_reward(self, td, actions) -> TensorDict:
    # Every step has a reward of -1
    step_count = td["step_count"]
    # if any of the batch element has reached maximum steps, set to infinity
    step_mask = step_count >= 100
    # give a reward of the step count, but inf for the non-done elements
    reward = torch.where(step_mask, torch.tensor(-1000.0, dtype=torch.float32, device=step_count.device), -step_count.float())
    # if this batch element has reached maximum steps, 
    # print("reward: ", reward)
    return reward

def _make_spec(self, td_params):
    """Make the observation and action specs from the parameters"""
    self.observation_spec = CompositeSpec(
        locs=BoundedTensorSpec(
            low=self.min_loc,
            high=self.max_loc,
            shape=(self.num_loc, 2),
            dtype=torch.float32,
        ),
        edges=UnboundedDiscreteTensorSpec(
            shape=(self.num_loc, self.num_loc),
            dtype=torch.bool,
        ),
        manhattan_matrix=UnboundedContinuousTensorSpec(
            shape=(self.num_loc, self.num_loc),
            dtype=torch.float32,
        ),
        start_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        end_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        current_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        i=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        action_mask=UnboundedDiscreteTensorSpec(
            shape=(self.num_loc),
            dtype=torch.bool,
        ),
        done_mask=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.bool,
        ),
        step_count=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        shape=(),
    )
    self.action_spec = BoundedTensorSpec(
        shape=(1,),
        dtype=torch.int64,
        low=0,
        high=self.num_loc,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
    self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

def generate_data(self, batch_size) -> TensorDict:
    # Ensure batch_size is an integer
    batch_size = int(batch_size[0]) if isinstance(batch_size, list) else batch_size

    grid_size = int(self.num_loc ** (1/2))

    # Generate normalize locations for the nodes
    grid_indices = torch.arange(grid_size)
    x, y = torch.meshgrid(grid_indices, grid_indices, indexing="ij")
    locs = torch.stack((x.flatten().float() / grid_size, y.flatten().float() / grid_size), dim=1)

    # Add batch dimension and repeat the locs tensor for each item in the batch
    locs = locs.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Generate the adjacency matrix
    edges = torch.zeros((batch_size, grid_size*grid_size, grid_size*grid_size), dtype=torch.bool)
    adjacency_matrix = generate_adjacency_matrix(grid_size)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.bool)
    adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    edges[:] = adjacency_matrix

    # print("locs: ", locs.shape)
    # print("edges: ", edges.shape)
    
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    
    return TensorDict({"locs": locs, "edges": edges}, batch_size=batch_size)

def plot_graph(self, locs, edges, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    locs = locs.detach().cpu()
    edges = edges.detach().cpu()

    x, y = locs[:, 0], locs[:, 1]

    # Plot the nodes
    ax.scatter(x, y, color="tab:blue")

    # Plot the edges
    x_i, y_i = locs[:, 0], locs[:, 1]
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                ax.plot([x_i[i], x_i[j]], [y_i[i], y_i[j]], color='g', alpha=0.1)

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    return ax

def generate_data_slow(self, batch_size) -> TensorDict:
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    
    num_loc = int(self.num_loc ** (1/2))
    
    # Generate random locations for the nodes
    locs = {}
    grid_size = num_loc
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / grid_size
            y = j / grid_size
            locs[(i, j)] = (x, y)
    locs = torch.tensor(list(locs.values()), dtype=torch.float32)
    locs = locs.unsqueeze(0).expand(batch_size + [-1, -1])
    # Generate a random adjaceny matrix for the edges
    edges = torch.zeros((*batch_size, num_loc, num_loc), dtype=torch.bool)
    for i in range(edges.shape[0]):
        matrix = generate_adjacency_matrix(grid_size)
        edges[i] = torch.tensor(matrix, dtype=torch.bool)
    return TensorDict({"locs": locs, "edges": edges}, batch_size=batch_size)

def render(self, td, actions=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]
        
    locs = td["locs"]

    # First and end node
    end_node = td["end_node"]
    x_end,  y_end= locs[end_node, 0], locs[end_node, 1]
    start_node = td["first_node"]
    x_start, y_start = locs[start_node, 0], locs[start_node, 1]
    
    # gather locs in order of action if available
    if actions is None:
        print("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        # Filter out the nodes after the end node
        end_idx = torch.where(actions == end_node)[0]
        if end_idx.numel() > 0:
            end_idx = end_idx[0]
            actions = actions[: end_idx + 1]
    
    a_locs = gather_by_index(locs, actions, dim=0)

    # Cat the start node to the start of the action locations
    a_locs = torch.cat([torch.tensor([[x_start, y_start]]), a_locs], dim=0)
    x, y = a_locs[:, 0], a_locs[:, 1]

    # Plot the visited nodes
    ax.scatter(locs[:, 0], locs[:, 1], color="tab:blue")

    # print("end node: ", end_node)
    # Highlight the start node in green
    ax.scatter(x[0], y[0], color="tab:green")

    # Highlight the end node in red
    ax.scatter(x[-1], y[-1], color="tab:red")

    # Plot the edges
    edges = td["edges"]
    x_i, y_i = locs[:, 0], locs[:, 1]
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                ax.plot([x_i[i], x_i[j]], [y_i[i], y_i[j]], color='g', alpha=0.1)

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(
        x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="r", alpha=1.0
    )

    # Highlight the last action
    ax.scatter(x_end, y_end, color="tab:red", s=100, edgecolors="black", zorder=10)
    # Highlight the first action
    ax.scatter(x_start, y_start, color="tab:green", s=100, edgecolors="black", zorder=10)

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

class SPPDynamicEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment"""

    name = "tsp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self._make_spec(td_params)
        
    invalidated = None

    _reset = _reset
    _step = _step
    get_reward = get_reward
    # check_solution_validity = check_solution_validity
    get_action_mask = get_action_mask
    _make_spec = _make_spec
    generate_data = generate_data
    render = render