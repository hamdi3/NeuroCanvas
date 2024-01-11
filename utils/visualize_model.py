import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from utils.generate_code import create_nn_code

def draw_graph():
    color_map = {
        'Linear': 'lightblue',
        'Bilinear': 'deepskyblue',
        'Conv2D': 'lightcoral',
        'ConvTranspose2d': 'indianred',
        'MaxPool2d': 'lightgreen',
        'AvgPool2d': 'darkseagreen',
        'BatchNorm2d': 'plum',
        'GroupNorm': 'mediumorchid',
        'Dropout': 'lightyellow',
        'Embedding': 'lightsalmon',
        'ReLU': 'lightpink',
        'Sigmoid': 'peachpuff',
        'Softmax': 'mistyrose'
    }

    # Get the colors for the nodes
    node_colors = [color_map[st.session_state.G.nodes[i]['label']]
                   for i in st.session_state.G.nodes]

    # Initialize the position dictionary
    pos = {}

    # Set the initial position
    x, y = 0, 0

    # Loop over the nodes
    for i in range(len(st.session_state.G.nodes)):
        # Add the current position to the dictionary
        pos[i] = (x, y)

        # Update the position
        if i % 12 < 5:  # First 4 nodes move up
            y += 1
        elif i % 12 < 6:  # Next 2 nodes move right
            x += 1
        elif i % 12 < 11:  # Next 4 nodes move down
            y -= 1
        else:  # Next 2 nodes move right
            x += 1

    # Draw the graph using Matplotlib
    plt.figure(figsize=(10, 8))
    node_labels = {
        i: st.session_state.G.nodes[i]['label'] for i in st.session_state.G.nodes}

    nx.draw(st.session_state.G, pos, with_labels=True, labels=node_labels,
            node_color=node_colors, node_size=5500, edge_color='gray', linewidths=1, font_size=10, font_color='black')

    st.pyplot(plt.gcf())

def visualize_model():
    # Display the network configuration
    tab1, tab2, tab3 = st.tabs(["Layers Table", "Layers Graph", "Generated Code"])
    with tab1:
        st.table(st.session_state["layers"])
    with tab2:
        draw_graph()
    with tab3:
        create_nn_code(st.session_state["layers"])