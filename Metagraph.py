import networkx as nx
import torch
import torch as t
import torch.nn as nn
from typing import Type

class MetaGraph(nx.Graph):

    def __init__(self, representation_size):
        super(MetaGraph, self).__init__()
        # https://stackoverflow.com/questions/29720222/inheriting-networkx-graph-and-using-nx-connected-component-subgraphs
        self.mg = nx.Graph()
        self.meta_vertices = [] # list of nx.graph
        self.across_edges = []
        self.vertex_id = 1
        self.meta_vertex_id = 1
        self.representation_size = representation_size
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.meta_vertex_representations = t.ones((self.meta_vertex_id, respresentation_size, respresentation_size))

    def add_vertex(self, nodes_representations):
        for node in nodes_representations:
            # check calc, for now assume it's right
            closest_concept_id = torch.argmax(self.cos(node.repeat(self.meta_vertex_id, 1, 1),
                                                       self.meta_vertex_representations))
            n = Node(self.vertex_id, closest_concept_id)
            self.meta_vertices[closest_concept_id].add_node(n)
            self.vertex_id += 1
            self.update_meta_vertex_representations(closest_concept_id, node)

    def add_meta_vertex(self):
        self.meta_vertices.append(nx.Graph())
        n = Node(self.meta_vertex_id, len(self.meta_vertices))
        self.mg.add_node(n)

    def add_edge(self,  node_a, node_b):
        if node_a.g_id == node_b.g_id:
            self.meta_vertices[node_a.g_id].add_edge(node_a.v_id, node_b.v_id)
        else:
            e = Edge(node_a.g_id, node_b.g_id, node_a.v_id, node_b.v_id)
            if len(self.across_edges) < max(node_a.v_id, node_b.v_id):
                self.__resize_matrix__(self.across_edges_matrix, max(node_a.v_id, node_b.v_id))
            # arranging based on g_id would be better for searching
            self.across_edges[node_a.v_id].append(e)
            self.across_edges[node_b.v_id].append(e)

    def add_meta_edge(self, node_a, node_b):
        self.mg.add_edge(node_a, node_b)

    def update_meta_vertex_representations(self, meta_vertex_id, node_representation):
        meta_g = self.meta_vertices[meta_vertex_id]
        no_of_nodes = len(meta_g.nodes())
        self.meta_vertex_representations[meta_vertex_id] *= (no_of_nodes-1)
        self.meta_vertex_representations[meta_vertex_id] += node_representation
        self.meta_vertex_representations[meta_vertex_id] /= no_of_nodes

    def find_unique_connections(self):
        # find vv: V in MV that connects with another vv that doesn't have any connections from other V in that same MV
        pass

    def get_rule_for_class(self):
        pass

    def get_replay_for_class(self):
        pass

    def get_transfer_material(self):
        pass

    def explain_prediction(self):
        pass

    def train(self):
        pass

    def __resize_list__(self, list, size):
        pass


class Node:
    def __init__(self, v_id, g_id):
        self.v_id = v_id
        self.g_id = g_id

class Edge:
    def __init__(self, g1_id, g2_id, v1_id, v2_id):
        self.g1_id, self.g2_id, self.v1_id, self.v2_id = g1_id, g2_id, v1_id, v2_id


# mg = MetaGraph(respresentation_size=11, no_of_concepts=64)
# mg.add_vertex([t.rand(11, 11)])
# print(mg.g)
node = Node(1, 1)
print(type(node))
