import torch.nn as nn
import torch
from rl4co.utils.ops import gather_by_index

class SPPInitEmbedding(nn.Module):
    """Initial embedding for the Shortest Path Problem (SPP) environment.
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes
        - target: x, y coordinates of the target node
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(SPPInitEmbedding, self).__init__()
        node_dim = 3 # x,y and distance_to_target
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

    def forward(self, td):
        locs = td["locs"]
        # print("locs: ", locs.shape)
        target = td["end_node"]
        #print("target: ", target.shape)

        batch_indices = torch.arange(len(target))
        target = locs[batch_indices, target].unsqueeze(1)

        #print("target: ", target.shape)
        #print(target)

        # Compute the distance to the target
        distance_to_target = torch.norm(locs - target, dim=-1, keepdim=True)
        # print("distance_to_target: ", distance_to_target.shape)
        #print(distance_to_target)
        
        # edges = td["edges"]

        # print("edges: ", edges.shape)
 
        # concat the edges to every node
        # edges = edges.unsqueeze(1).expand(-1, locs.shape[1], -1, -1)
        # print("edges: ", edges.shape)
        
        # locs_and_edges = torch.cat([locs, edges], dim=2)
        #print("locs_and_edges: ", locs_and_edges.shape)

        locs_and_distance = torch.cat([locs, distance_to_target], dim=2)

        # locs_and_distances_and_edges = torch.cat([locs, distance_to_target, edges], dim=2)
        # Get neighbor indices
        # neighbor_indices = edges.nonzero(as_tuple=True)[1].reshape(locs.shape[1], -1)

        # # Get neighbor embeddings
        # neighbor_embeddings = self.init_embed(locs[:, neighbor_indices])

        # # Aggregate neighbor embeddings
        # aggregated_neighbor_embeddings = neighbor_embeddings.sum(dim=2)

        # # Get node embeddings
        # node_embeddings = self.init_embed(locs)

        # # Concatenate node embeddings and aggregated neighbor embeddings
        # out = torch.cat([node_embeddings, aggregated_neighbor_embeddings], dim=2)
        # print("out: ", out)
        # Concatenate the locs and target to form the input tensor
        #node_and_target = torch.cat([locs, target], dim=2)
        out = self.init_embed(locs_and_distance)

        return out
    
class SPPContext(nn.Module):
    """Context embedding for the Shortest Path Problem (SPP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
        - target node embedding
    """

    def __init__(self, embedding_dim,  linear_bias=True):
        super(SPPContext, self).__init__()
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * embedding_dim).uniform_(-1, 1)
        )
        self.project_context = nn.Linear(
            embedding_dim*2, embedding_dim, bias=linear_bias
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["end_node"].dim() == 1 else (td["end_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["end_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)
    
class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0