import streamlit as st
import networkx as nx

from utils.menu import render_menu
from utils.visualize_model import visualize_model

# Define page config
st.set_page_config(page_title="NeuroCanvas", page_icon="🎨", layout="wide", initial_sidebar_state="auto",
                   menu_items={
                    'Report a bug': "https://github.com/hamdi3/NeuroCanvas/issues",
                    'About': "# 🎨 :rainbow[NeuroCanvas]\n This is an app made to help you easily **Create**, **visualize** and **Train** neuronal networks. \n \n Any suggestions are welcomed [here]('https://github.com/hamdi3/NeuroCanvas/issues')."
                    })

# Initialize an empty list to store the layers
if "layers" and "G" not in st.session_state:
    st.session_state["layers"] = []
    st.session_state["G"] = nx.DiGraph()


def main():
    # For centering the title
    _,colT2 = st.columns([1,2.2])
    with colT2:
        st.title("🎨 :rainbow[NeuroCanvas]")

    # Render the selection menu
    render_menu()

    # Visualize the model
    visualize_model()

if __name__ == "__main__":
    main()
