import networkx as nx
import matplotlib.pyplot as plt
from itertools import product


G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3)])

# print all vertices
#print("Vertices of graph:")
#print(list(product(G.nodes(), repeat=3)))

#print("\nEdges of graph:")
#print(G.edges())

colour_dict = {}
k = 2

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

def update_colours(atomic_vector_list, colour_dict={}):
    all_tuples = list(product(G.nodes(), repeat=k))
    number_of_tuples = len(all_tuples)
    for i in range(number_of_tuples):
        colour_dict[all_tuples[i]] = get_tuple_hash(atomic_vector_list[i])

    return colour_dict


def initialise(atomic_vector_list = []):
    all_tuples = list(product(G.nodes(), repeat=k))
    number_of_tuples = len(all_tuples)

    for i in range(number_of_tuples):
        atomic_vector_list.append(get_atomic_type_vector(all_tuples[i], G))

    update_colours(atomic_vector_list, colour_dict)



    return atomic_vector_list, colour_dict




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


def k_wl_algorithm(G, k):
    # --- Step 0: Initialization ---
    all_k_tuples = list(product(G.nodes(), repeat=k))
    
    # Initial color map (C^0) based on the matrix K hash (your first step)
    # Since we are focusing on the refinement loop, we assume an initial coloring exists.
    # For a true k-WL, this would be based on the matrix K.
    
    # We use a placeholder here, assuming all tuples start with color 1.
    current_coloring = {t: 1 for t in all_k_tuples}
    
    # Store the previous set of unique colors to check for convergence
    prev_num_unique_colors = 0 
    
    # Loop variables
    iteration = 1
    max_iterations = 100 # Safety stop

    while iteration <= max_iterations:
        print(f"\n--- Starting Iteration {iteration} ---")
        
 
        raw_signatures = refine(G, k, current_coloring) 
        
        new_color_list, signature_map, num_unique_colors = get_new_colors(raw_signatures)

        print(f"Number of unique colors found: {num_unique_colors}")

        
        if num_unique_colors == prev_num_unique_colors:
            print(f" CONVERGED at Iteration {iteration}. No new structural patterns found.")
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


# 1. Compute the Initial Coloring (C^0)
atomic_vector_list, initial_colouring = initialise()

print("Initial C^0 Coloring computed.")

# 2. Run the k-WL Algorithm
final_features = k_wl_algorithm(G, k)

# 3. Print Final Result
print("\n--- FINAL K-WL FEATURE VECTOR ---")
print(f"Total Unique Stable Colors: {len(final_features)}")
print(f"Color Histogram (Feature Vector): {final_features}")