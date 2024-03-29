import streamlit as st
import networkx as nx

from utils.codes import execute_code
from utils.menu import render_menu
from utils.visualizer import visualize_model, visualize_model_preformance, visualize_trainer

# Define page config
st.set_page_config(page_title="NeuroCanvas", page_icon="🎨", layout="wide", initial_sidebar_state="auto",
                   menu_items={
                    'Report a bug': "https://github.com/hamdi3/NeuroCanvas/issues",
                    'About': "# 🎨 :rainbow[NeuroCanvas]\n This is an app made to help you easily **Create**, **visualize** and **Train** neuronal networks. \n \n Any suggestions are welcomed [here]('https://github.com/hamdi3/NeuroCanvas/issues')."
                    })

# Initialize an empty list to store the layers
if "layers" and "G" not in st.session_state:
    st.session_state["layers"] = []
    st.session_state["trainer"] = []
    st.session_state["G"] = nx.DiGraph()


def main():
    # For centering the title
    _,colT2 = st.columns([1,2.2])
    with colT2:
        st.title("🎨 :rainbow[NeuroCanvas]")

    # Render the selection menu
    render_menu()

    st.header(":blue[Model]")
    with st.expander("Show Model Section",True):
        # Visualize the model
        visualize_model()

    st.header(":orange[Trainer]")
    with st.expander("Show Trainer Section",True):
        # Visualize the trainer
        visualize_trainer()

    if st.button(f'Train Model'):
        st.header("Training Outputs")
        tab1, tab2 = st.tabs(["Code Output", "Model Preformance"])
        with tab1:
            execute_code()
        with tab2:
            visualize_model_preformance()

if __name__ == "__main__":
    main()
