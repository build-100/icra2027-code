import copy
import math
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import torch


def _d_opt(matrix):
    """Return det(matrix) ** (1 / n) for a positive-definite matrix."""
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("The information matrix must be positive definite.")
    return float(np.exp(logdet / matrix.shape[0]))


def _stable_sorted(values):
    """Sort homogeneous labels naturally, with a deterministic fallback."""
    values = list(values)
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _add_edge_information_matrix(graph, information):
    for edge in graph.edges():
        graph.edges[edge]["information"] = information.copy()


def _add_graph_weights_as_dopt(graph, key="d_opt"):
    for edge in graph.edges():
        graph.edges[edge][key] = _d_opt(graph.edges[edge]["information"])

class PriorGraph:
    def __init__(self, path=None, width=1.0, start_node=None, device="cpu", graph=None):
        if graph is None and path is None:
            raise ValueError("Either an XML path prefix or a NetworkX graph is required.")

        self.xml_path = None if path is None else str(path) + ".xml"
        self.map_width = width
        self.start_node = start_node
        self.need_noise = False
        self.variance = 0
        self.prior_graph = (
            graph.copy()
            if graph is not None
            else self.build_prior_graph_with_xml(
                need_normalize=False,
                need_noise=self.need_noise,
                variance=self.variance,
            )
        )
        # self.prior_graph = nx.Graph()
        # self.prior_graph.add_node(0, position=(0,0))
        # self.prior_graph.add_node(1, position=(1,0))
        # self.prior_graph.add_node(2, position=(2,0))
        # self.prior_graph.add_node(3, position=(2,1))
        # self.prior_graph.add_node(4, position=(1,1))
        # self.prior_graph.add_node(5, position=(0,1))
        # self.prior_graph.add_node(6, position=(0,2))
        # self.prior_graph.add_node(7, position=(1,2))
        # self.prior_graph.add_node(8, position=(2,2))

        # self.prior_graph.add_edge(0, 1)
        # self.prior_graph.add_edge(1, 2)

        # self.prior_graph.add_edge(2, 3)
        # self.prior_graph.add_edge(3, 4)
        # self.prior_graph.add_edge(4, 5)
        # self.prior_graph.add_edge(0, 5)
        # self.prior_graph.add_edge(1, 4)

        # self.prior_graph.add_edge(5, 6)
        # self.prior_graph.add_edge(6, 7)
        # self.prior_graph.add_edge(7, 8)
        # self.prior_graph.add_edge(4, 7)
        # self.prior_graph.add_edge(3, 8)
        if any("weight" not in self.prior_graph.edges[edge] for edge in self.prior_graph.edges()):
            self.add_edge_distance_as_weight(self.prior_graph)
        self.update_prior_graph_attributes()    # edge weight: distance      information: imformation matrix     d-opt: d-opt
        self.num_nodes = self.prior_graph.number_of_nodes()
        if self.num_nodes == 0:
            raise ValueError("The prior graph must contain at least one planning node.")
        self.node_labels = _stable_sorted(self.prior_graph.nodes())
        if self.start_node is None:
            self.start_node = self.node_labels[0]
        if self.start_node not in self.prior_graph:
            raise ValueError(f"Unknown start node: {self.start_node!r}")
        self.node_to_index = {node: index for index, node in enumerate(self.node_labels)}
        self.index_to_node = {index: node for node, index in self.node_to_index.items()}
        self.neighbors_by_node = {
            node: _stable_sorted(self.prior_graph.neighbors(node)) for node in self.node_labels
        }
        self.embedding_layer = None
        self.create_list(device)
        self.reset(self.start_node, device)
        print(
            "Prior graph has {} nodes and {} edges".format(
                self.prior_graph.number_of_nodes(), self.prior_graph.number_of_edges()
            )
        )
        # self.exact_tsp_solver()
        # self.TSP_path = [3, 11, 12, 6, 5, 4, 2, 7, 8, 9, 10, 13, 14, 15, 21, 22, 23, 24, 20, 19, 18, 17, 16]
        # print("TSP path:{}".format(self.TSP_path))
        
    def create_list(self, device):
        """Materialize one deterministic action ordering for the whole runtime."""
        device = torch.device(device)
        self.adj_list = [self.neighbors_by_node[node] for node in self.node_labels]
        self.edge_distance = [
            [self.prior_graph.edges[node, neighbor]["weight"] for neighbor in neighbors]
            for node, neighbors in zip(self.node_labels, self.adj_list)
        ]
        self.edge_dopt = [
            [self.prior_graph.edges[node, neighbor]["d_opt"] for neighbor in neighbors]
            for node, neighbors in zip(self.node_labels, self.adj_list)
        ]
        self.max_degree = max((len(neighbors) for neighbors in self.adj_list), default=0)
        # A singleton graph still uses a one-wide Q head, with its only slot masked.
        self.action_dim = max(1, self.max_degree)

        self.adj_tensor = torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.long, device=device)
        self.distance_tensor = torch.zeros((self.num_nodes, self.max_degree), device=device)
        self.dopt_tensor = torch.zeros((self.num_nodes, self.max_degree), device=device)
        for index, neighbors in enumerate(self.adj_list):
            distances = self.edge_distance[index]
            dopts = self.edge_dopt[index]
            neighbor_indices = [self.node_to_index[node] for node in neighbors]
            self.adj_tensor[index, :len(neighbors)] = torch.tensor(
                neighbor_indices, dtype=torch.long, device=device
            )
            self.distance_tensor[index, :len(distances)] = torch.tensor(
                distances, dtype=torch.float32, device=device
            )
            self.dopt_tensor[index, :len(dopts)] = torch.tensor(
                dopts, dtype=torch.float32, device=device
            )

    def valid_neighbors(self, node):
        if node not in self.neighbors_by_node:
            raise KeyError(f"Unknown node: {node!r}")
        return self.neighbors_by_node[node]

    def action_mask(self, node, device=None):
        """Return a mask where True denotes an invalid padded action."""
        mask = torch.ones(self.action_dim, dtype=torch.bool, device=device)
        mask[: len(self.valid_neighbors(node))] = False
        return mask

    def action_to_node(self, node, action):
        action_index = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        neighbors = self.valid_neighbors(node)
        if action_index < 0 or action_index >= len(neighbors):
            raise ValueError(f"Action {action_index} is invalid at node {node!r}.")
        return neighbors[action_index]

    def action_for_neighbor(self, node, neighbor):
        try:
            return self.valid_neighbors(node).index(neighbor)
        except ValueError as error:
            raise ValueError(f"{neighbor!r} is not adjacent to {node!r}.") from error

    def action_schema(self):
        return {
            "node_labels": list(self.node_labels),
            "action_dim": self.action_dim,
            "neighbors": [list(self.neighbors_by_node[node]) for node in self.node_labels],
            # Checkpoint compatibility must include reward dynamics, not topology alone.
            "edge_distances": [list(distances) for distances in self.edge_distance],
            "edge_dopts": [list(dopts) for dopts in self.edge_dopt],
        }

    def build_prior_graph_with_xml(self, need_normalize = False, need_noise=False, variance=0) -> nx.graph:
        graph = self.build_prior_map_from_drawio(self.xml_path, actual_map_width=self.map_width, 
                                            need_normalize=need_normalize, need_noise=need_noise, variance=variance)
        # rospy.loginfo(f"Build prior map with {len(graph.nodes())} vertices.")
        return graph
    
    def update_prior_graph_attributes(self):
        # Add attributes for path planning.  An explicit scalar ``omega`` (or
        # legacy ``d_opt``) is authoritative for the GEVD graph interface;
        # otherwise retain the historical isotropic information default.
        Cov = np.zeros((3, 3))
        Cov[0, 0] = 0.1
        Cov[1, 1] = 0.1
        Cov[2, 2] = 0.001
        Sigma = np.linalg.inv(Cov)  # information matrix
        for edge in self.prior_graph.edges():
            attributes = self.prior_graph.edges[edge]
            information = np.asarray(
                attributes.get("information", Sigma), dtype=np.float64
            )
            attributes["information"] = information.copy()
            if "omega" in attributes:
                omega = float(attributes["omega"])
            elif "d_opt" in attributes:
                omega = float(attributes["d_opt"])
            else:
                omega = _d_opt(information)
            distance = float(attributes["weight"])
            if not np.isfinite(omega) or omega <= 0.0:
                raise ValueError(f"Edge {edge!r} must have omega > 0.")
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError(f"Edge {edge!r} must have distance >= 0.")
            attributes["omega"] = omega
            attributes["d_opt"] = omega
        return

    def from_drawio_to_nx(self, file_name: str) -> nx.graph:
        """ Given a xml file exported from drawio, traverse it into a networkx graph object.
        """
        # Read XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        nodes_dict = {}
        edge_list = []

        # Traverse XML tree
        for element in root.iter():
            # Read the element tag and attributes.
            tag = element.tag
            if tag != "mxCell":
                continue

            # print(f"Current tag: {tag}")
            attributes = element.attrib
            if "style" not in attributes:
                continue
            if attributes["style"][:7] == "ellipse":  # node
                new_node = {}
                geometry_element = element.find("mxGeometry")
                geometry_attributes = geometry_element.attrib
                for key in ["x", "y"]:
                    if key in geometry_attributes:
                        new_node[key] = float(geometry_attributes[key])
                    else:
                        new_node[key] = 0
                nodes_dict[attributes["id"]] = new_node
            elif attributes["style"][:9] == "edgeStyle":   # edge
                source = attributes["source"]
                target = attributes["target"]
                edge_list.append([source, target])

        graph = nx.Graph()
        id_to_node = {}
        node_index = 0
        for id, node_dict in nodes_dict.items():
            x, y = node_dict["x"], node_dict["y"]
            graph.add_node(node_index, position = (x, y))
            id_to_node[id] = node_index
            node_index += 1

        for source, target in edge_list:
            source_node = id_to_node[source]
            target_node = id_to_node[target]
            graph.add_edge(source_node, target_node)

        return graph

    def move_graph_to_border(self, graph, actual_width = -1):
        """                   mid_node
            left_node                         right_node
        """
        isolated_nodes_list = list(nx.isolates(graph))
        isolated_nodes = [graph.nodes()[node]["position"] for node in isolated_nodes_list]
        left_idx, right_index, mid_index = -1, -1, -1
        # 012, 021, 102, 120, 201, 210
        possible = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        index = -1
        if isolated_nodes[0][0] < isolated_nodes[1][0] and isolated_nodes[1][0] < isolated_nodes[2][0]:
            index = 0
        elif isolated_nodes[0][0] < isolated_nodes[2][0] and isolated_nodes[2][0] < isolated_nodes[1][0]:
            index = 1
        elif isolated_nodes[1][0] < isolated_nodes[0][0] and isolated_nodes[0][0] < isolated_nodes[2][0]:
            index = 2
        elif isolated_nodes[1][0] < isolated_nodes[2][0] and isolated_nodes[2][0] < isolated_nodes[0][0]:
            index = 3
        elif isolated_nodes[2][0] < isolated_nodes[0][0] and isolated_nodes[0][0] < isolated_nodes[1][0]:
            index = 4
        elif isolated_nodes[2][0] < isolated_nodes[1][0] and isolated_nodes[1][0] < isolated_nodes[0][0]:
            index = 5
        left_idx, mid_index, right_index = possible[index]
        left_node, mid_node, right_node = isolated_nodes[left_idx], isolated_nodes[mid_index], isolated_nodes[right_index]
        # print([left_node, mid_node, right_node])

        width_in_drawio = right_node[0] - left_node[0]
        if actual_width < 0:
            scale = 1
        else:
            scale = actual_width / width_in_drawio
        # print(f"Drawio scale: {scale}")  # 0.1028 for map3
        
        center_x_drawio, center_y_drawio = 0.5 * (left_node[0] + right_node[0]), 0.5 * (left_node[1] + mid_node[1])
        # print([center_x_drawio, center_y_drawio])

        # Shift the graph so that it is centered at the mid_node, and scaled by the "scale"
        # Step 1: for all points, x = x - center_x_drawio, y = y - center_y_drawio
        # Step 2: for all points, y = -y;
        # Step 3: for all points, x = x * scale, y = y * scale
        for node in graph.nodes():
            x, y = graph.nodes()[node]["position"]
            new_x = x - center_x_drawio
            new_y = y - center_y_drawio
            new_y *= -1
            new_x *= scale
            new_y *= scale
            graph.nodes()[node]["position"] = (new_x, new_y)
            
        return

    def remove_isolated_nodes(self, graph):
        """ Remove nodes for positioning the graph. """
        isolated_nodes_list = list(nx.isolates(graph))
        for node in isolated_nodes_list:
            graph.remove_node(node)
        return

    def add_edge_distance_as_weight(self, graph):
        """ Add distance metric as weight in the graph. """
        for edge in graph.edges():
            source, target = edge
            pos1, pos2 = graph.nodes()[source]["position"], graph.nodes()[target]["position"]
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            graph.edges()[edge]["weight"] = distance
        return

    def build_prior_map_from_drawio(self, path: str, actual_map_width: float = -1, need_normalize: bool = False, 
                                    need_noise: bool = False, variance: float = 0) -> nx.graph:
        """ Given a xml file for a drawio graph, transform it into a networkx graph. 
            The node name may not start from 0, becuase isolated nodes are removed.
        """
        graph = self.from_drawio_to_nx(path)
        # Add noise to the position parameters
        if need_noise:
            for node in graph.nodes():
                x, y = graph.nodes()[node]["position"]
                graph.nodes()[node]["position"] = (x + np.random.normal(0, variance), y + np.random.normal(0, variance))
        # Relocate the graph
        self.move_graph_to_border(graph, actual_width = actual_map_width)
        self.remove_isolated_nodes(graph)
        self.add_edge_distance_as_weight(graph)

        # Test: normalize the edge distance
        if need_normalize:
            min_distance = float("inf")
            for edge in graph.edges():
                min_distance = min(min_distance, graph.edges()[edge]["weight"])
            if min_distance > 0:
                for edge in graph.edges():
                    prev_distance = graph.edges()[edge]["weight"]
                    divided = round(prev_distance / min_distance)
                    if divided == 0:
                        print("Normalize edge distance error!")
                        continue
                    graph.edges()[edge]["weight"] = divided * min_distance
        return graph
    







    def reset(self, start_node=None, device=None):
        if start_node is None:
            start_node = self.start_node
        if start_node not in self.prior_graph:
            raise ValueError(f"Unknown start node: {start_node!r}")
        self.state = self.initialize(start_node)
        self.start_node = start_node
        # Track episode termination; the start node is already visited in s_0.
        self.visited = {start_node}
        # Accumulate the trajectory reward.
        self.path = nx.Graph()
        self.laplacian_fixed = np.eye(self.num_nodes)
        self.old_inv = np.eye(self.num_nodes)
        self.old_dopt = 0.0
        self.long_distance = 0.0
        self.loop = []
        # Preserve the active legacy trainer's s_0 feature convention while the
        # internal accumulator remains at zero for the first information update.
        return self.encode(self.state, 1.0, 0)


    def initialize(self, start_node):
        return start_node
    
    def step(self, action, state=None, first_node=None, device=None, h=0):
        if state is not None:
            state_value = state.item() if isinstance(state, torch.Tensor) else state
            if state_value != self.state:
                raise ValueError(
                    f"Step state {state_value!r} does not match environment state {self.state!r}."
                )
        next_state = self.action_to_node(self.state, action)
        distance, marg, dopt = self.get_laplacian(
            self.state, next_state, self.num_nodes, h
        )
        self.state = next_state
        self.visited.add(self.state)
        next_traj = self.encode(self.state, dopt, h + 1)
        done = self.is_coverage_complete()
        return next_state, done, next_traj, distance, marg

    def is_coverage_complete(self):
        return len(self.visited) == self.num_nodes
    
    def _trajectory_to_observation(self, states):
        # Legacy helper: initialize lazily so the active DQN path has no hidden seed.
        if self.embedding_layer is None:
            from gevd.models.networks import EmbeddingLayer

            self.embedding_layer = EmbeddingLayer(self.num_nodes, 4, seed=None)
        clipped_trajectory = copy.copy(states)
        return self.embedding_layer(torch.tensor(clipped_trajectory))

    def get_laplacian(self, state, next_state, n, h):
        cur_index = self.node_to_index[state]
        next_index = self.node_to_index[next_state]
        action = self.action_for_neighbor(state, next_state)
        distance = float(self.distance_tensor[cur_index, action].item())
        dopt = float(self.dopt_tensor[cur_index, action].item())
        q = np.zeros(n)
        q[cur_index] = 1
        q[next_index] = -1
        q_inv_q = dopt * q.T @ self.old_inv @ q

        # Update the Laplacian matrix.
        if self.path.has_edge(state, next_state):
            marg = 0.0
        else:
            self.laplacian_fixed[cur_index, cur_index] += dopt
            self.laplacian_fixed[next_index, next_index] += dopt
            self.laplacian_fixed[cur_index, next_index] -= dopt
            self.laplacian_fixed[next_index, cur_index] -= dopt
            new_dopt = self.old_dopt +np.log(1+q_inv_q)/(n-1)
            marg = float(new_dopt - self.old_dopt)
            self.old_dopt = new_dopt
            new_inv = self.old_inv - (dopt * self.old_inv @ np.outer(q, q) @ self.old_inv) / (1+ q_inv_q)
            self.old_inv = new_inv

            self.long_distance += distance
            if next_state in self.visited and not self.path.has_edge(state, next_state):
                self.long_distance = 0
                self.loop.append(h+1)
            if self.long_distance >=250:
                marg -= 0.05

            self.path.add_edge(state, next_state)
            self.path.nodes[state]['position'] = self.prior_graph.nodes[state]['position']
            self.path.nodes[next_state]['position'] = self.prior_graph.nodes[next_state]['position']

        return distance, marg, self.old_dopt
    
    def get_agent_obs_onehot(self, node):
        obs_onehot = list(np.eye(self.num_nodes, dtype=np.float32)[self.node_to_index[node]])
        return obs_onehot
    
    def encode(self, node, dopt,h):
        visited_mask = [1 if label in self.visited else 0 for label in self.node_labels]
        state = self.get_agent_obs_onehot(node) + visited_mask + [dopt] + [h]
        return state

    def exact_tsp_solver(self):
        """
        Solve the travelling-salesperson problem exactly with dynamic programming.
        Return (shortest route, shortest distance).
        """
        nodes = list(self.prior_graph.nodes())
        n = len(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        
        # Build the adjacency dictionary and distance matrix.
        adj_dict = {node: set(self.prior_graph.neighbors(node)) for node in nodes}
        dist_matrix = [[math.inf] * n for _ in range(n)]
        for u, v in self.prior_graph.edges():
            i, j = node_index[u], node_index[v]
            dist_matrix[i][j] = dist_matrix[j][i] = self.prior_graph.edges[u, v]['weight']
        
        # Initialize the dynamic-programming table.
        dp = [[math.inf] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        # Initialize the starting state.
        start_idx = node_index[self.start_node]
        dp[1 << start_idx][start_idx] = 0
        
        # Populate the dynamic-programming table.
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == math.inf:
                    continue
                    
                # Consider only edges present in the graph.
                last_node = nodes[last]
                for neighbor in adj_dict[last_node]:
                    next_idx = node_index[neighbor]
                    if mask & (1 << next_idx):
                        continue
                        
                    new_mask = mask | (1 << next_idx)
                    new_dist = dp[mask][last] + dist_matrix[last][next_idx]
                    
                    if new_dist < dp[new_mask][next_idx]:
                        dp[new_mask][next_idx] = new_dist
                        parent[new_mask][next_idx] = last
        
        # Backtrack to recover the optimal route.
        full_mask = (1 << n) - 1
        min_dist = min(dp[full_mask])
        last_node_idx = dp[full_mask].index(min_dist)
        
        # Reconstruct the route.
        path = []
        mask = full_mask
        current = last_node_idx
        while current != -1:
            path.append(nodes[current])
            prev = parent[mask][current]
            if prev == -1:
                break
            mask ^= (1 << current)
            current = prev
        
        # Check that the route is valid.
        valid_path = True
        for i in range(len(path)-1):
            if path[i+1] not in adj_dict[path[i]]:
                valid_path = False
                break
        
        if not valid_path:
            raise ValueError("No valid travelling-salesperson route exists; the graph may be disconnected.")
        self.TSP_path = list(reversed(path))

    # def solve_tsp(self):
    #     """
    #     Approximate a travelling-salesperson route.
    #     Return the ordered route through all nodes.
    #     """
    #     # Use a simple nearest-neighbor heuristic.
    #     path = [self.start_node]
    #     unvisited = set(self.prior_graph.nodes()) - {self.start_node}
        
    #     while unvisited:
    #         last = path[-1]
    #         next_node = min(unvisited, key=lambda x: self.prior_graph.edges[last, x]['weight'])
    #         path.append(next_node)
    #         unvisited.remove(next_node)
        
    #     self.TSP_path = path
    



def get_laplacian(prior_graph: nx.Graph, path: nx.Graph, cur_node: int, next_node: int):
       # old_G_laplacian = G_laplacian

       # add edge to path and get d-opt and distance
       path.add_edge(cur_node, next_node)
       path.nodes()[cur_node]['position'] = prior_graph.nodes()[cur_node]['position']
       path.nodes()[next_node]['position'] = prior_graph.nodes()[next_node]['position']
       edge = (cur_node, next_node)
       D_opt = prior_graph.edges[edge]['d_opt']
       path.edges[edge]['d_opt'] = D_opt
       G_laplacian = nx.laplacian_matrix(path, weight='d_opt').toarray()
       return path, G_laplacian

def update_prior_graph_attributes(prior_graph):
    # Add attributes for path planning
    Cov = np.zeros((3, 3))
    Cov[0, 0] = 0.1
    Cov[1, 1] = 0.1
    Cov[2, 2] = 0.001
    Sigma = np.linalg.inv(Cov)  # information matrix
    _add_edge_information_matrix(prior_graph, Sigma)
    _add_graph_weights_as_dopt(prior_graph, key="d_opt")
    return prior_graph

def append_state(traj, H, device):
    cur_h = len(traj)
    traj = torch.vstack(traj)
    dummy = -1*torch.ones(H-cur_h, traj.shape[1], device=device)
    state_h = torch.cat([traj, dummy]).transpose(0, 1)
    return state_h

def show_graph(graph: nx.Graph):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for node in graph.nodes():
        x, y = graph.nodes()[node]["position"]
        ax.plot(x, y, 'o', markersize=10, color='g', alpha = 0.7)
        plt.text(x, y+0.1, str(node))
    for edge in graph.edges():
        node1, node2 = edge
        node1_pos, node2_pos = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
        ax.plot([node1_pos[0], node2_pos[0]], [node1_pos[1], node2_pos[1]], '-', color='r', alpha = 0.5, zorder=5)
    ax.set_aspect('equal')
    plt.show()

