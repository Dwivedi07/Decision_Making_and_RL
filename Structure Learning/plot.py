import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Read the .gph file and extract the edges
def read_gph(filename):
    edges = []
    with open(filename, 'r') as file:
        for line in file:
            nodes = line.strip().split(', ')
            edges.append((nodes[0], nodes[1]))
    return edges

# Step 2: Create a directed graph and add edges
def create_and_plot_graph(edges):
    G = nx.DiGraph()  # Create a directed graph
    G.add_edges_from(edges)  # Add edges to the graph

    # Step 3: Draw the graph using networkx and matplotlib
    pos = nx.circular_layout(G)  # Positions for all nodes
    plt.figure(figsize=(8, 6))

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=400, arrows=True, font_size=11, font_weight='normal', edge_color='black')
    
    # Show the graph
    plt.title("Directed Graph Visualization")
    plt.show()

# Step 4: Main function to run the steps
if __name__ == "__main__":
    filename = 'large11111.gph'  # The name of your .gph file
    edges = read_gph(filename)  # Read edges from the file
    create_and_plot_graph(edges)  # Create and plot the graph
