import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict, Tuple, List
import matplotlib.colors as mcolors

# --- 0. GRAPH SETUP ---
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 2), (3, 6), (1, 5)])
k = 2 

# --- UTILITY: DRAWING FUNCTION ---

def get_node_coloring_from_k_tuples(G: nx.Graph, k: int, coloring: Dict[Tuple[int], int]) -> Dict[int, int]:
    """
    Derives a color for each single node (1-tuple) based on the colors of the 
    k-tuples it belongs to. We use the color of the TUPLE (v, v, ..., v) for simplicity.
    """
    node_colors = {}
    for node in G.nodes():
        # Create the k-tuple where all elements are the current node (e.g., (2, 2) for node 2)
        v_tuple = tuple([node] * k)
        
        # Use the stable color of this special k-tuple as the node's color
        # This is a common simplification for visualizing k-WL node equivalence
        if v_tuple in coloring:
             node_colors[node] = coloring[v_tuple]
        else:
             node_colors[node] = 0 # Default if not found
             
    return node_colors

def draw_coloring(G: nx.Graph, k: int, coloring: Dict[Tuple[int], int], iteration: int):
    """Draws the graph, using derived colors for the nodes."""
    
    # 1. Derive single node colors from the k-tuple coloring
    node_color_map = get_node_coloring_from_k_tuples(G, k, coloring)
    
    # 2. Prepare plot
    plt.figure(figsize=(4, 4))
    
    # Use spring_layout for a consistent look
    pos = nx.spring_layout(G, seed=42) 
    
    # Get the list of colors in node order
    node_color_list = [node_color_map[node] for node in G.nodes()]
    
    # Normalize the colors to a color map (needed for non-sequential integer colors)
    unique_colors = sorted(list(set(node_color_list)))
    
    # Create a distinct color for each unique color value (e.g., 1, 2, 3...)
    # We use a colormap like tab10 for clear separation
    cmap = plt.cm.get_cmap('tab10', max(1, len(unique_colors)))
    
    # Map the integer color (1, 2, 3) to an actual hex color
    color_map_indices = [unique_colors.index(c) for c in node_color_list]
    node_hex_colors = [mcolors.rgb2hex(cmap(i)) for i in color_map_indices]

    # 3. Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_hex_colors, node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    
    # Add labels showing the node ID and its current color
    node_labels = {node: f'{node}\n(C: {node_color_map[node]})' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black')
    
    plt.title(f"k-WL Iteration {iteration} (k={k})\nUnique Node Colors: {len(unique_colors)}")
    plt.axis('off')
    plt.show()


# --- 1. INITIALIZATION FUNCTIONS (C^0) ---
# (These remain unchanged from your last code)
def get_atomic_type_vector(v_tuple: Tuple[int], graph: nx.Graph) -> List[int]:
    k_len = len(v_tuple)
    atomic_type_vector = []
    for i in range(k_len):
        v_i = v_tuple[i]
        for j in range(k_len):
            v_j = v_tuple[j]
            if v_i == v_j:
                atomic_type_vector.append(2)
            elif graph.has_edge(v_i, v_j):
                atomic_type_vector.append(1)
            else:
                atomic_type_vector.append(0)
    return atomic_type_vector

def get_tuple_hash(atomic_type_vector: List[int]) -> int:
    return hash(tuple(atomic_type_vector))

def initialise(graph: nx.Graph, k: int) -> Dict[Tuple[int], int]:
    all_tuples = list(product(graph.nodes(), repeat=k))
    colour_dict = {}
    for v_tuple in all_tuples:
        vector = get_atomic_type_vector(v_tuple, graph)
        colour_dict[v_tuple] = get_tuple_hash(vector)
    return colour_dict

# --- 2. REFINEMENT FUNCTION ---
def refine(G: nx.Graph, k: int, colour_dict: Dict[Tuple[int], int]) -> List[List[int]]:
    nodes = list(G.nodes())
    all_raw_signatures = []
    for current_tuple, old_color in colour_dict.items():
        signature_list = [old_color] 
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

# --- 3. CANONICALIZATION FUNCTION ---
def get_new_colors(raw_signatures: List[List[int]]) -> Tuple[List[int], Dict[Tuple, int], int]:
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

# --- 4. MAIN WL ALGORITHM LOOP ---

def k_wl_algorithm(G: nx.Graph, k: int, initial_coloring: Dict[Tuple, int]) -> Dict[int, int]:
    """Runs the k-WL algorithm, plotting node colors at each step."""
    all_k_tuples = list(product(G.nodes(), repeat=k))
    current_coloring = initial_coloring
    prev_num_unique_colors = 0 
    
    iteration = 0 # Start iteration count at 0 for C^0
    max_iterations = 10 

    # Plot Initial Coloring (C^0)
    num_unique_colors = len(set(initial_coloring.values()))
    draw_coloring(G, k, current_coloring, iteration)
    prev_num_unique_colors = num_unique_colors
    
    iteration += 1

    while iteration <= max_iterations:
        print(f"\n--- Starting Iteration {iteration} ---")
        
        # 1. REFINEMENT
        raw_signatures = refine(G, k, current_coloring) 
        
        # 2. CANONICALIZATION
        new_color_list, _, num_unique_colors = get_new_colors(raw_signatures)

        # 3. Preparation for Next Iteration (Update C^t)
        new_coloring_dict = {}
        for i, t in enumerate(all_k_tuples):
            new_coloring_dict[t] = new_color_list[i]
            
        current_coloring = new_coloring_dict
        
        # 4. Draw Current Coloring (C^t)
        draw_coloring(G, k, current_coloring, iteration)

        # 5. CONVERGENCE CHECK
        print(f"Number of unique k-tuple colors: {num_unique_colors}")
        if num_unique_colors == prev_num_unique_colors:
            print(f"✅ CONVERGED at Iteration {iteration}. Structure is stable.")
            break
            
        prev_num_unique_colors = num_unique_colors
        iteration += 1

    # Final Step: Extract Histogram (Features)
    final_histogram = {}
    for color in current_coloring.values():
        final_histogram[color] = final_histogram.get(color, 0) + 1
        
    return final_histogram

# --- EXECUTION ---

# 1. Compute the Initial Coloring (C^0)
initial_coloring = initialise(G, k)

print(f"Initial C^0 Coloring computed. Total unique k-tuple hashes: {len(set(initial_coloring.values()))}")

# 2. Run the k-WL Algorithm and plot steps
final_features = k_wl_algorithm(G, k, initial_coloring)

# 3. Print Final Result
print("\n--- FINAL K-WL FEATURE VECTOR ---")
print(f"Final Stable Colors: {len(final_features)}")
print(f"Color Histogram (Feature Vector): {final_features}")