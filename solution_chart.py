import networkx as nx
from pyvis.network import Network

# Create a directed graph
G = nx.DiGraph()

# Define nodes (columns) and their relationships (you will need to adjust these based on actual relationships)
nodes = ['Data Collection', 'Data Cleansing', 'Feature Engineering', 'Model Training', 'Model Evaluation', 'Post Processing']
edges = [
    ('Data Collection', 'Data Cleansing'),
    ('Data Cleansing', 'Feature Engineering'),
    ('Feature Engineering', 'Model Training'),
    ('Model Training', 'Model Evaluation'),
    ('Model Evaluation', 'Post Processing')
]

# Add nodes and edges to the graph
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Generate a network graph with physics controls using pyvis
nt = Network('500px', '1000px', notebook=False)
nt.from_nx(G)
nt.show_buttons(filter_=['physics'])  # Enable physics controls
nt.enable_physics(True)  # Turn on physics for interactive exploration

# Save the graph to an HTML file
html_file_path = 'network_graph.html'
nt.show(html_file_path)
