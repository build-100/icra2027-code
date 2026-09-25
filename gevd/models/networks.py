
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence


class QNetwork(nn.Module):
    """Small feed-forward action-value network that returns raw Q values."""

    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        if obs_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim, hidden_dim, and action_dim must be positive.")
        self.q_values = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(weights_init_)

    def forward(self, observation):
        return self.q_values(observation)


class GraphUtilityNetwork(nn.Module):
    """Configurable graph-aware local utility network with shared parameters.

    Dynamic inputs contain own history plus broadcast visit/representative flags.
    Static prior-graph tensors are registered buffers and are therefore shared by
    No teammate traversal history, current pose, or relative metric transform is
    supplied. Broadcast identity channels use self then cyclic robot order.
    """

    def __init__(
        self,
        num_nodes,
        num_edges,
        action_dim,
        edge_endpoints,
        edge_static_features,
        neighbor_nodes,
        neighbor_edges,
        hidden_dim,
        message_passing_layers=1,
        residual_updates=False,
        normalize_local_features=False,
        t_max=None,
        initialization="legacy_gain_0_1",
        num_robots=1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_edges = int(num_edges)
        self.action_dim = int(action_dim)
        self.num_robots = int(num_robots)
        if self.num_robots < 1:
            raise ValueError("num_robots must be positive.")
        self.broadcast_channels = 1 + 2 * self.num_robots
        self.obs_dim = self.num_edges + (2 + self.broadcast_channels) * self.num_nodes + 1
        hidden_dim = int(hidden_dim)
        self.message_passing_layers = int(message_passing_layers)
        self.residual_updates = bool(residual_updates)
        self.normalize_local_features = bool(normalize_local_features)
        self.t_max = None if t_max is None else int(t_max)
        self.initialization = str(initialization)
        if min(self.num_nodes, self.action_dim, hidden_dim) <= 0 or self.num_edges < 0:
            raise ValueError("Invalid graph utility dimensions.")
        if self.message_passing_layers < 1:
            raise ValueError("message_passing_layers must be positive.")
        if self.normalize_local_features and (self.t_max is None or self.t_max < 1):
            raise ValueError("A positive t_max is required for feature normalization.")
        if self.initialization not in {"legacy_gain_0_1", "relu_orthogonal"}:
            raise ValueError("Unsupported graph-utility initialization.")

        endpoints = torch.as_tensor(edge_endpoints, dtype=torch.long).reshape(
            self.num_edges, 2
        )
        edge_static = torch.as_tensor(
            edge_static_features, dtype=torch.float32
        ).reshape(self.num_edges, 2)
        neighbor_nodes = torch.as_tensor(neighbor_nodes, dtype=torch.long).reshape(
            self.num_nodes, self.action_dim
        )
        neighbor_edges = torch.as_tensor(neighbor_edges, dtype=torch.long).reshape(
            self.num_nodes, self.action_dim
        )
        if self.num_edges:
            if (endpoints < 0).any() or (endpoints >= self.num_nodes).any():
                raise ValueError("Edge endpoints are outside the graph node range.")
        valid_slots = neighbor_nodes >= 0
        if not torch.equal(valid_slots, neighbor_edges >= 0):
            raise ValueError("Neighbor-node and neighbor-edge padding must agree.")
        if valid_slots.any():
            if (neighbor_nodes[valid_slots] >= self.num_nodes).any():
                raise ValueError("Neighbor index is outside the graph node range.")
            if (neighbor_edges[valid_slots] >= self.num_edges).any():
                raise ValueError("Neighbor edge index is outside the edge range.")

        degree = torch.zeros(self.num_nodes, dtype=torch.float32)
        if self.num_edges:
            degree.index_add_(0, endpoints[:, 0], torch.ones(self.num_edges))
            degree.index_add_(0, endpoints[:, 1], torch.ones(self.num_edges))
        self.register_buffer("edge_endpoints", endpoints)
        self.register_buffer("edge_static_features", edge_static)
        self.register_buffer("neighbor_nodes", neighbor_nodes)
        self.register_buffer("neighbor_edges", neighbor_edges)
        self.register_buffer("node_degree", degree.clamp_min(1.0))

        self.node_encoder = nn.Sequential(
            nn.Linear(3 + self.broadcast_channels, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
        )
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.extra_messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(self.message_passing_layers - 1)
        )
        self.extra_node_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(self.message_passing_layers - 1)
        )
        self.residual_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim)
            for _ in range(self.message_passing_layers)
        ) if self.residual_updates else nn.ModuleList()
        self.utility_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.initialization == "legacy_gain_0_1":
            self.apply(weights_init_)
        else:
            self.apply(relu_orthogonal_init_)
            torch.nn.init.orthogonal_(self.utility_head[-1].weight, gain=1.0)
            torch.nn.init.zeros_(self.utility_head[-1].bias)

    def forward(self, observation):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected local observations shaped [batch, {self.obs_dim}]."
            )
        batch_size = observation.shape[0]
        traversals = observation[:, : self.num_edges]
        visited_start = self.num_edges
        visited = observation[:, visited_start : visited_start + self.num_nodes]
        current = observation[
            :, visited_start + self.num_nodes : visited_start + 2 * self.num_nodes
        ]
        time_value = observation[:, -1:]
        if self.normalize_local_features:
            scale = float(self.t_max)
            traversals = traversals / scale
            time_value = time_value / scale

        own_features = torch.stack(
            (
                visited,
                current,
                time_value.expand(-1, self.num_nodes),
            ),
            dim=-1,
        )
        metadata_start = visited_start + 2 * self.num_nodes
        metadata = observation[:, metadata_start:-1].reshape(
            batch_size, self.broadcast_channels, self.num_nodes
        ).transpose(1, 2)
        node_features = torch.cat((own_features, metadata), dim=-1)
        node_base = self.node_encoder(node_features)

        static_edges = self.edge_static_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        edge_features = torch.cat((traversals.unsqueeze(-1), static_edges), dim=-1)
        edge_embedding = self.edge_encoder(edge_features)

        node_embedding = node_base
        message_layers = (self.message, *self.extra_messages)
        update_layers = (self.node_update, *self.extra_node_updates)
        for layer_index, (message_layer, update_layer) in enumerate(
            zip(message_layers, update_layers)
        ):
            aggregate = torch.zeros_like(node_embedding)
            if self.num_edges:
                left = self.edge_endpoints[:, 0]
                right = self.edge_endpoints[:, 1]
                message_to_left = message_layer(
                    torch.cat((node_embedding[:, right], edge_embedding), dim=-1)
                )
                message_to_right = message_layer(
                    torch.cat((node_embedding[:, left], edge_embedding), dim=-1)
                )
                aggregate.index_add_(1, left, message_to_left)
                aggregate.index_add_(1, right, message_to_right)
            aggregate = aggregate / self.node_degree.view(1, -1, 1)
            updated = update_layer(torch.cat((node_embedding, aggregate), dim=-1))
            node_embedding = (
                self.residual_norms[layer_index](node_embedding + updated)
                if self.residual_updates
                else updated
            )

        current_index = current.argmax(dim=1)
        batch_index = torch.arange(batch_size, device=observation.device)
        current_embedding = node_embedding[batch_index, current_index]
        graph_embedding = node_embedding.mean(dim=1)

        neighbor_nodes = self.neighbor_nodes[current_index]
        neighbor_edges = self.neighbor_edges[current_index]
        safe_nodes = neighbor_nodes.clamp_min(0)
        safe_edges = neighbor_edges.clamp_min(0)
        neighbor_embedding = node_embedding[
            batch_index.unsqueeze(1), safe_nodes
        ]
        if self.num_edges:
            action_edge_embedding = edge_embedding[
                batch_index.unsqueeze(1), safe_edges
            ]
        else:
            action_edge_embedding = torch.zeros_like(neighbor_embedding)
        current_expanded = current_embedding.unsqueeze(1).expand(
            -1, self.action_dim, -1
        )
        graph_expanded = graph_embedding.unsqueeze(1).expand(
            -1, self.action_dim, -1
        )
        utility_input = torch.cat(
            (
                current_expanded,
                neighbor_embedding,
                action_edge_embedding,
                graph_expanded,
            ),
            dim=-1,
        )
        return self.utility_head(utility_input).squeeze(-1)


class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                #    nn.Dropout(p=0.3),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim))
        self.apply(weights_init_)

    def forward(self, x, action_mask):
        logits = self.actor(x)
        logits_masked = logits.masked_fill(action_mask, float('-inf'))
        return F.softmax(logits_masked, dim=-1)
    
    def create_action_masks(self, total_actions, valid_actions_list, device):
        action_masks = []
        for valid_actions in valid_actions_list:
            action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
            action_masks.append(action_mask)
        action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool).to(device)
        return action_masks_tensor




class CriticNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    # nn.Dropout(p=0.3),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        self.apply(weights_init_)
    
    def forward(self, x):
        return self.critic(x)



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, seed=None):
        super(EmbeddingLayer, self).__init__()
        if seed is not None:
            set_seed(seed)
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.embedding(x)


# class Policy(torch.nn.Module):
#     def __init__(self, obs_dim, hidden_dim, action_dim):
#         super(Policy, self).__init__()
#         self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim)

#     def forward(self, x, action_mask):
#         lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
#         x = pad_sequence(x, batch_first=True)
#         mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
#         mask = mask.to(x.device)
#         x, _ = self.gru(x)
#         x = x * mask.unsqueeze(2).float()
#         x = x[torch.arange(x.size(0)), lengths - 1]
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x)
#         logits_masked = logits.masked_fill(action_mask, float('-inf'))
#         return F.softmax(logits_masked, dim=1)

# class Value(torch.nn.Module):
#     def __init__(self, obs_dim, hidden_dim, action_dim):
#         super(Value, self).__init__()
#         self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, x, action_mask):
#         lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
#         x = pad_sequence(x, batch_first=True)
#         mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
#         mask = mask.to(x.device)
#         x, _ = self.gru(x)
#         x = x * mask.unsqueeze(2).float()
#         x = x[torch.arange(x.size(0)), lengths - 1]
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x




def create_action_masks(total_actions, valid_actions_list, device):
    action_masks = []
    for valid_actions in valid_actions_list:
        action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
        action_masks.append(action_mask)
    action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool).to(device)
    return action_masks_tensor

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=0.1)
        # torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.01)


def relu_orthogonal_init_(module):
    """Stable non-degenerate initialization for configurable graph encoders."""

    if isinstance(module, nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
        torch.nn.init.zeros_(module.bias)
