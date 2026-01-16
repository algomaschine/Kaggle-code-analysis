import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def tokenize_filename(filename):
    tokens = re.split('[-_ ]', filename)  # Splitting by '-', '_', and space
    return tokens

def build_token_graph(directory):
    token_graph = nx.Graph()
    token_count = Counter()
    co_occurrence = defaultdict(int)

    # Walk through the directory and process each file
    for _, _, files in os.walk(directory):
        for file in files:
            tokens = tokenize_filename(file)
            token_count.update(tokens)
            # Update co-occurrence graph
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    if tokens[i] != tokens[j]:  # Ensure we're connecting different tokens
                        co_occurrence[(tokens[i], tokens[j])] += 1

    # Add nodes and edges based on token frequencies and co-occurrences
    for token, freq in token_count.items():
        token_graph.add_node(token, size=freq)

    for (token1, token2), freq in co_occurrence.items():
        token_graph.add_edge(token1, token2, weight=freq)

    return token_graph

def draw_token_graph(graph):
    pos = nx.spring_layout(graph)  # positions for all nodes
    sizes = [graph.nodes[node]['size'] * 100 for node in graph]  # Scale node size
    nx.draw(graph, pos, with_labels=True, node_size=sizes,
            node_color='skyblue', font_size=8, font_weight='bold',
            edge_color='gray')
    plt.title("Token Co-occurrence in Filenames")
    plt.show()

if __name__ == "__main__":
    import re
    directory = 'path_to_your_directory'
    token_graph = build_token_graph(directory)
    draw_token_graph(token_graph)
