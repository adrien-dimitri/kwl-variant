import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np 
import random # Added for setting random seed for consistent layout

# --- Visualization Helpers ---

def map_k_tuple_colors_to_nodes(G, k, coloring):
    """
    Maps the k-tuple colors to node colors using the diagonal (v, v, ..., v) tuple
    as a proxy for the node's structural role.
    
    Returns: node_color_map {node: unique_color_index}, num_unique_colors
    """
    node_colors = {}
    unique_colors = sorted(list(set(coloring.values())))
    num_unique_colors = len(unique_colors)
    
    # Create a mapping from k-tuple color ID to a simplified 0-based index for colormap
    color_id_to_index = {color_id: i for i, color_id in enumerate(unique_colors)}

    for node in G.nodes():
        # Get the color of the diagonal tuple (v, v, ..., v)
        diagonal_tuple = tuple([node] * k)
        
        # Get the k-tuple color ID (which is a hash)
        color_id = coloring.get(diagonal_tuple, 0) 
        
        # Use the mapped index for plotting
        node_colors[node] = color_id_to_index.get(color_id, 0)
        
    return node_colors, num_unique_colors

def draw_graph_coloring(G, coloring_dict, iteration, k, pos):
    """
    Draws the graph, coloring nodes based on the k-tuple coloring's diagonal element.
    """
    
    # 1. Map k-tuple colors to node colors
    node_color_map, num_unique_colors = map_k_tuple_colors_to_nodes(G, k, coloring_dict)
    
    # 2. Assign Matplotlib colors
    if num_unique_colors == 0:
        mpl_colors = ['gray'] * len(G.nodes())
        legend_text = f"Iter {iteration}: No unique colors found"
    else:
        # Use a distinguishable colormap (e.g., tab20)
        cmap = plt.cm.get_cmap('tab20', max(1, num_unique_colors))
        
        # Get the list of colors in node order for networkx drawing
        node_order = list(G.nodes())
        # The color mapping index must be normalized by the total number of unique colors
        mpl_colors = [cmap(node_color_map[node] / max(1, num_unique_colors)) for node in node_order]
        
        legend_text = f"Iter {iteration}: {num_unique_colors} unique colors"

    plt.figure(figsize=(8, 8))
    
    nx.draw_networkx_nodes(G, pos, node_color=mpl_colors, node_size=700, alpha=0.9, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
    
    plt.title(f"k-WL Coloring (k={k}) - {legend_text}")
    plt.text(0.5, 0.95, f"Total Nodes (N): {len(G.nodes())}", transform=plt.gcf().transFigure, ha='center', fontsize=10, color='gray')

    # Print the current coloring distribution to the console
    print(f"\n[Visualizer] Iteration {iteration}: {num_unique_colors} Unique k-tuple Colors")
    
    plt.axis('off')
    plt.show()

# --- Core Algorithm Functions ---

def generate_erdos_renyi_graph(num_nodes, prob):
    """Generates an Erdos-Renyi graph."""
    G = nx.erdos_renyi_graph(num_nodes, prob)
    return G

def get_atomic_type_vector(v_tuple, graph):
    """
    Computes the atomic type vector (0=no edge, 1=edge, 2=self-loop/same node) 
    for a k-tuple of nodes.
    """
    k = len(v_tuple)
    atomic_type_vector = []

    # build atomic type vector: checks all pairs (v_i, v_j)
    for i in range(len(v_tuple)):
        v_i = v_tuple[i]
        for j in range(len(v_tuple)):
            v_j = v_tuple[j]

            if v_i == v_j:
                atomic_type_vector.append(2)
            elif graph.has_edge(v_i, v_j):
                atomic_type_vector.append(1)
            else:
                atomic_type_vector.append(0)

    return atomic_type_vector

def get_tuple_hash(atomic_type_vector):
    """Returns a hash for a list/tuple of integers."""
    return hash(tuple(atomic_type_vector))

def initialise(graph, k):
    """
    Performs the 0-WL step: initializes the color for every k-tuple based on 
    its atomic type vector.
    """
    all_tuples = list(product(graph.nodes(), repeat=k))
    colour_dict = {}

    for v_tuple in all_tuples:
        vector = get_atomic_type_vector(v_tuple, graph)
        colour_dict[v_tuple] = get_tuple_hash(vector)

    return colour_dict


def refine(G, k, colour_dict):
    """
    Standard k-WL refinement step. The multiset includes the color of a 
    tuple where one element is replaced by *any* node l in V.
    """
    nodes = list(G.nodes())
    all_raw_signatures = []
    for current_tuple, old_colour in colour_dict.items():
        signature_list = [old_colour]
        
        # Check all k positions (j)
        for j in range(k):
            multiset = []
            # Check all nodes (node_l)
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
    """
    Variant k-WL refinement step. The multiset only includes the color of a 
    tuple where one element v_j is replaced by a *neighbor* w of v_j.
    """
    all_raw_signatures = []
    for current_tuple, old_color in colour_dict.items():
        signature_list = [old_color]
        for j in range(k):
            v_j = current_tuple[j]
            multiset = []
            
            # Iterate only over NEIGHBORS (w in N(v_j))
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
    """Maps raw signatures to canonical new color integers."""
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


def k_wl_algorithm(G, k, initial_colouring, variant=False, pos=None):
    """
    The main k-WL iterative algorithm. Includes visualization calls.
    Requires 'pos' (networkx layout) for consistent drawing.
    """
    all_k_tuples = list(product(G.nodes(), repeat=k))
    current_coloring = initial_colouring.copy()
    
    # Store the previous set of unique colors to check for convergence
    prev_num_unique_colors = len(set(current_coloring.values())) 
    
    # --- Visualization: Initial Coloring (Iteration 0) ---
    if pos is not None:
        draw_graph_coloring(G, current_coloring, 0, k, pos)
    
    # Loop variables
    iteration = 1
    max_iterations = 100 

    while iteration <= max_iterations:
        
        if variant:
            raw_signatures = refine_variant(G, k, current_coloring)
        else:
            raw_signatures = refine(G, k, current_coloring)

        new_color_list, _, num_unique_colors = get_new_colors(raw_signatures)
        
        
        # --- Convergence Check ---
        if num_unique_colors == prev_num_unique_colors:
            print(f"[Visualizer] CONVERGED at Iteration {iteration}. No new structural patterns found.")
            break
        
        # --- Update Coloring ---
        new_coloring_dict = {}
        for i, t in enumerate(all_k_tuples):
            new_coloring_dict[t] = new_color_list[i]
            
        current_coloring = new_coloring_dict
        prev_num_unique_colors = num_unique_colors
        
        # --- Visualization: Refinement Step ---
        if pos is not None:
            draw_graph_coloring(G, current_coloring, iteration, k, pos)
        
        iteration += 1

    final_histogram = {}
    for color in current_coloring.values():
        final_histogram[color] = final_histogram.get(color, 0) + 1
        
    return final_histogram


# --- Main Visualization Execution ---
if __name__ == "__main__":
    # --- Parameters for Visualization ---
    num_nodes = 10 # Keep N small for clear visualization
    prob = 0.3     # Edge probability
    k = 2          # k-WL parameter
    
    random.seed(42) # Set seed for random graph
    
    print(f"--- Starting k-WL Visualization (N={num_nodes}, k={k}, P={prob}) ---")

    # 1. Generate Graph
    G = generate_erdos_renyi_graph(num_nodes, prob)
    
    # Check if the graph has any edges (for a meaningful k-WL)
    if not G.edges:
        print("\nWarning: Generated graph is empty. Re-generating graph...")
        # Try a fixed probability until we get a non-empty graph (max 5 tries)
        for _ in range(5):
            G = generate_erdos_renyi_graph(num_nodes, prob)
            if G.edges:
                break
        if not G.edges:
            print("Failed to generate a non-empty graph. Exiting visualization.")
        
    
    # 2. Compute fixed layout (crucial for consistent plot sequence)
    pos = nx.spring_layout(G, seed=42)

    # 3. Run Standard k-WL with visualization
    print("\n[Running Standard k-WL]")
    initial_colouring_standard = initialise(G, k)
    k_wl_algorithm(G, k, initial_colouring_standard, variant=False, pos=pos)
    
    # 4. Run Variant k-WL with visualization (Optional: uncomment to run)
    # print("\n[Running Variant k-WL]")
    # initial_colouring_variant = initialise(G, k)
    # k_wl_algorithm(G, k, initial_colouring_variant, variant=True, pos=pos)
