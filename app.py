import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="NeuroCanvas", page_icon="🧊", layout="wide", initial_sidebar_state="auto", menu_items=None)

    # Initialize an empty list to store the layers
    if "layers" and "G" not in st.session_state:
        st.session_state["layers"] = []
        st.session_state["G"] = nx.DiGraph()

    # Define the categories
    categories = {
        'Main Layers': ['Conv2D', 'Dense'],
        'Regulation Layers': ['MaxPooling2D', 'Dropout']
        # Add more categories as needed
    }

    # Initialize the layer parameters dictionary
    layer_params = {}

    # Loop over the categories
    for category, layers in categories.items():
        # Create a dropdown for the current category
        layer_type = st.sidebar.selectbox(f'Select the layer type ({category})', layers)

        # Configure the parameters based on the selected layer type
        if layer_type == 'Conv2D':
            layer_params['filters'] = st.sidebar.number_input('Number of filters', value=32)
            layer_params['kernel_size'] = st.sidebar.number_input('Kernel size', value=3)
            # Add more parameters as needed
        elif layer_type == 'Dense':
            layer_params['units'] = st.sidebar.number_input('Number of units', value=32)
            # Add more parameters as needed
        # Add more elif conditions for other layer types

        # Add the layer when the button is clicked
        if st.sidebar.button(f'Add {layer_type} layer'):
            # Add the configured layer to the network
            st.session_state.layers.append({
                'Layer Type': layer_type,
                'Parameters': ', '.join(f'{key.replace("_"," ")}: {value}' for key, value in layer_params.items())
            })

            # Add a node to the graph for the new layer
            st.session_state.G.add_node(len(st.session_state.layers) - 1, label=layer_type)

            # If this is not the first layer, add an edge from the previous layer
            if len(st.session_state.layers) > 1:
                st.session_state.G.add_edge(len(st.session_state.layers) - 2, len(st.session_state.layers) - 1)

    if st.sidebar.button('Delete last layer',type="primary"):
        # Delete the last added layer
        if st.session_state["layers"] and st.session_state["G"]:
            last_node = len(st.session_state.layers) - 1
            st.session_state.G.remove_node(last_node)
            st.session_state.layers.pop()

    # Main
    st.title("Neural Network Configuration")
    # Display the network configuration
    with st.expander("Show Layers Table", expanded=True):
        st.table(st.session_state["layers"])

    color_map = {
        'Conv2D': 'lightcoral',
        'Dense': 'lightblue',
        'MaxPooling2D': 'lightgreen',
        'Dropout': 'lightyellow'
        # Add more colors for other layer types
    }

    # Get the colors for the nodes
    node_colors = [color_map[st.session_state.G.nodes[i]['label']] for i in st.session_state.G.nodes]

    # Initialize the position dictionary
    pos = {}

    # Set the initial position
    x, y = 0, 0

    # Loop over the nodes
    for i in range(len(st.session_state.G.nodes)):
        # Add the current position to the dictionary
        pos[i] = (x, y)

        print(i)
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
    with st.expander("Visualize Network", expanded=True):
        plt.figure(figsize=(10, 8))
        node_labels = {i: st.session_state.G.nodes[i]['label'] for i in st.session_state.G.nodes}
        
        nx.draw(st.session_state.G, pos, with_labels=True, labels=node_labels,
                node_color=node_colors, node_size=5500, edge_color='gray', linewidths=1, font_size=10, font_color='black')
        
        st.pyplot(plt.gcf())

    if st.button('Train model'):
        # Train the model
        st.write('Training the model...')

if __name__ == "__main__":
    main()
