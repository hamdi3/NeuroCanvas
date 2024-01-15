import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

from utils.codes import create_nn_code, generate_trainer_code

COLOR_MAP = {
    'Linear': 'lightblue',
    'Bilinear': 'deepskyblue',
    'Conv2d': 'lightcoral',
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


def draw_graph():
    # Get the colors for the nodes
    node_colors = [COLOR_MAP[st.session_state.G.nodes[i]['label']]
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

def visualize_code(code_string):
    st.code(f"""{code_string}""", language="python")

def visualize_model():
    # Display the network configuration
    tab1, tab2, tab3 = st.tabs(
        ["Layers Table", "Layers Graph", "Generated Code"])
    with tab1:
        st.table(st.session_state["layers"])
    with tab2:
        draw_graph()
    with tab3:
        visualize_code(create_nn_code(st.session_state["layers"]))


def visualize_trainer():
    tab1, tab2 = st.tabs(["Component Table", "Generated Code"])
    with tab1:
        st.table(st.session_state["trainer"])
    with tab2:
        trainer_code, _ = generate_trainer_code(st.session_state["trainer"])
        visualize_code(trainer_code)

def visualize_model_preformance():
    model = torch.jit.load('model_scripted.pt')  # Load the scripted model
    model.eval()  # Set the model to evaluation mode

    # Loading the testing dataset
    for comp in st.session_state["trainer"]:
        if comp['Component'] == 'MNIST':
            test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        elif comp['Component'] == 'FashionMNIST':
            test_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    # Visualizing the graph
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 3, 4

    for i in range(1, cols * rows + 1):
        # Making the prediction
        sample_idx = torch.randint(len(test_data), size=(1,)).item() 
        img, label = test_data[sample_idx]

        # Pass the input data through the model
        output = model(img.view(img.shape[0], -1))

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the class with the highest probability
        _, pred = torch.max(probabilities, dim=1)

        # Adding the subplots
        figure.add_subplot(rows, cols, i)
        plt.title(f"Pred: {pred.item()}, Label: {label}", fontdict={"fontsize": 14, "color": ("green" if pred == label else "red")})
        plt.axis("off")
        plt.imshow(img.cpu().squeeze()) 

    st.pyplot(figure)