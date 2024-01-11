import streamlit as st
import networkx as nx

from utils.menu import render_menu
from utils.visualize_model import visualize_model

# Define page config
st.set_page_config(page_title="NeuroCanvas", page_icon="🎨"
                  , layout="wide", initial_sidebar_state="auto")

# Initialize an empty list to store the layers
if "layers" and "G" not in st.session_state:
    st.session_state["layers"] = []
    st.session_state["G"] = nx.DiGraph()


def main():
    # Render the selection menu
    render_menu()

    # Visualize the model
    visualize_model()

if __name__ == "__main__":
    main()
