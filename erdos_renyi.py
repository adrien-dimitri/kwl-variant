# generate erdos renyi graph
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np


def generate_erdos_renyi_graph(num_nodes, prob):
    G = nx.erdos_renyi_graph(num_nodes, prob)
    return G


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

def initialise(graph, k):
    all_tuples = list(product(graph.nodes(), repeat=k))
    colour_dict = {}

    for v_tuple in all_tuples:
        vector = get_atomic_type_vector(v_tuple, graph)
        colour_dict[v_tuple] = get_tuple_hash(vector)

    return colour_dict


def refine(G, k, colour_dict):
    nodes = list(G.nodes())
    all_raw_signatures = []
    for current_tuple, old_colour in colour_dict.items():
        signature_list = [old_colour]
        for j in range(k):
            multiset = []
            for node_l in nodes: 
                temp_tuple_list = list(current_tuple)
                temp_tuple_list[j] = node_l 
                neighbor_tuple = tuple(temp_tuple_list)
                
                neighbor_color = colour_dict.get(neighbor_tuple)
                
                if neighbor_color is not None:
                    multiset.append(neighbor_color)


            multiset.sort()

            multiset_hash = hash(tuple(multiset))

            signature_list.append(multiset_hash)

        all_raw_signatures.append(signature_list)

    return all_raw_signatures

def refine_variant(G, k, colour_dict):
    all_raw_signatures = []
    for current_tuple, old_color in colour_dict.items():
        signature_list = [old_color]
        for j in range(k):
            v_j = current_tuple[j]
            multiset = []
            
            # Iterate only over NEIGHBORS (N(v_j))
            for w in G.neighbors(v_j): 
                temp_tuple_list = list(current_tuple)
                temp_tuple_list[j] = w
                neighbor_tuple = tuple(temp_tuple_list)

                neighbor_color = colour_dict.get(neighbor_tuple)
                if neighbor_color is not None:
                    multiset.append(neighbor_color)
            
            multiset.sort()
            multiset_hash = hash(tuple(multiset))
            signature_list.append(multiset_hash) 
            
        all_raw_signatures.append(signature_list)
    return all_raw_signatures

def get_new_colors(raw_signatures):
    signature_tuples = [tuple(s) for s in raw_signatures]

    signature_to_new_color = {}
    new_color_counter = 1
    new_color_list = []

    for signature in signature_tuples:
        if signature not in signature_to_new_color:
            signature_to_new_color[signature] = new_color_counter
            new_color_counter += 1

        new_color_list.append(signature_to_new_color[signature])

    num_unique_colors = new_color_counter - 1
    
    return new_color_list, signature_to_new_color, num_unique_colors


def k_wl_algorithm(G, k, initial_colouring, variant=False):
    # --- Step 0: Initialization ---
    all_k_tuples = list(product(G.nodes(), repeat=k))
    
    current_coloring = initial_colouring.copy()
    
    # Store the previous set of unique colors to check for convergence
    prev_num_unique_colors = len(set(current_coloring.values())) 
    
    # Loop variables
    iteration = 1
    max_iterations = 100 # Safety stop

    while iteration <= max_iterations:
        if variant:
            raw_signatures = refine_variant(G, k, current_coloring)
        else:
            raw_signatures = refine(G, k, current_coloring)

        new_color_list, signature_map, num_unique_colors = get_new_colors(raw_signatures)

        
        if num_unique_colors == prev_num_unique_colors:
            break
        

        new_coloring_dict = {}
        for i, t in enumerate(all_k_tuples):
            new_coloring_dict[t] = new_color_list[i]
            
        current_coloring = new_coloring_dict
        prev_num_unique_colors = num_unique_colors
        
        iteration += 1

    final_histogram = {}
    for color in current_coloring.values():
        final_histogram[color] = final_histogram.get(color, 0) + 1
        
    return final_histogram


if __name__ == "__main__":
    # Example usage
    num_nodes = 25
    prob = 0.05
    k = 2

    # start
    p_start = 0
    p_end = 0.99 
    p_step = 0.01
    p_values = [round(x, 2) for x in np.arange(p_start, p_end + p_step, p_step)]

    standard_time = []
    variant_time = []

    for prob in p_values:
        G = generate_erdos_renyi_graph(num_nodes, prob)
        standard_start_time = time.time()
        initial_colouring = initialise(G, k)
        final_features_standard = k_wl_algorithm(G, k, initial_colouring)
        standard_time.append(time.time() - standard_start_time)

        variant_start_time = time.time()
        final_features_variant = k_wl_algorithm(G, k, initial_colouring, variant=True)
        variant_time.append(time.time() - variant_start_time)

    # Plot results against the probability values
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, standard_time, label=f'Standard k-WL (N={num_nodes}, k={k})', color='blue', marker='o', markersize=3, linewidth=1)
    plt.plot(p_values, variant_time, label=f'Variant k-WL (N={num_nodes}, k={k})', color='red', marker='x', markersize=3, linewidth=1)
    
    plt.xlabel('Edge Probability (P)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'k-WL Execution Time on Erdős-Rényi Graphs (N={num_nodes}, k={k})')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()


