import networkx as nx
import torch
import torch as t
import torch.nn as nn

class MetaGraph(nx.Graph):

    def __init__(self, respresentation_size, no_of_concepts):
        super(MetaGraph, self).__init__()
        # https://stackoverflow.com/questions/29720222/inheriting-networkx-graph-and-using-nx-connected-component-subgraphs
        self.mg = nx.Graph()
        self.vertices = []
        self.meta_vertices = [] #list of nx.graph
        self.edges = []
        self.meta_edges = []
        self.vertex_id = 1
        self.meta_vertex_id = 1
        self.respresentation_size = respresentation_size
        # self.no_of_concepts = no_of_concepts
        # fixing number of concepts now, can make dynamic later with extend
        # self.concept_vertices = t.ones((no_of_concepts, respresentation_size, respresentation_size))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.meta_vertex_representations = t.ones((self.meta_vertex_id, respresentation_size, respresentation_size))
        self.edges_across_mv = []

    def add_vertex(self, nodes_representations):
        for node in nodes_representations:
            closest_concept_id = torch.argmax(self.cos(node.repeat(self.meta_vertex_id, 1, 1),
                                                       self.meta_vertex_representations))
            self.meta_vertices[closest_concept_id].add_node(self.vertex_id)
            self.vertex_id += 1
            self.update_meta_vertex_representations(closest_concept_id, node)

    def add_meta_vertex(self):
        self.meta_vertices.append(nx.Graph())
        self.mg.add_node(self.meta_vertex_id)

    def add_edge(self,  node_a, node_b):
        pass

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




mg = MetaGraph(respresentation_size=11, no_of_concepts=64)
mg.add_vertex([t.rand(11, 11)])
print(mg.g)
