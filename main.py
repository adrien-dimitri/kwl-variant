import networkx as nx
import matplotlib.pyplot as plt
from itertools import product


G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4)])

# print all vertices
#print("Vertices of graph:")
#print(list(product(G.nodes(), repeat=3)))

#print("\nEdges of graph:")
#print(G.edges())

def get_atomic_type_vector(v_tuple, graph):
    
    k = len(v_tuple)

    atomic_type_vector = []

    # build atomic type vector
    for i in range(len(v_tuple)):
        v_i = v_tuple[i]
        for j in range(len(v_tuple)):
            v_j = v_tuple[j]

            if v_i == v_j:
                atomic_type_vector.append((2))

            elif graph.has_edge(v_i, v_j):
                atomic_type_vector.append((1))

            else:
                atomic_type_vector.append((0))

    return atomic_type_vector

def get_tuple_hash(atomic_type_vector):
    return hash(tuple(atomic_type_vector))

v_tuple_example = (1, 3, 2) # A 3-tuple
atomic_vector = get_atomic_type_vector(v_tuple_example, G)
final_hash = get_tuple_hash(atomic_vector)

print(f"Tuple: {v_tuple_example}")
print(f"Atomic Vector (K matrix rows concatenated): {atomic_vector}")
print(f"Final Hash: {final_hash}")