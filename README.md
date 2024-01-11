# NeuroCanvas

[![Apache-2.0 License](https://img.shields.io/badge/license-Apache2.0-blue.svg)](https://github.com/hamdi3/NeuroCanvas/blob/main/LICENSE)
[![Python 3.10.11](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Streamlit 1.29.0](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
![Built with Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red.svg)


Welcome to the NeuroCanvas GitHub repository! This project aims to help those with little to no coding knoweldge to create and visualize neuronal networks in an easy user-friendly way.

## Features
- Through the menu on the right the user can select between the model and the trainer setting.
- When the menu settings are selected the user then can easily add layers or activations to the model
  - When a layer is added the view on the right side of the page automatically updates.
  - The user can view the model as a table or a directed graph.
  - Additionally a code is generated for the made model.
- Same thing goes for the trainer menu except no graph.
  - Keeping in mind that the user can only add one optimizer and one dataset to avoid any issues with the generated code.  
- Deployed web application accessible at [https://neurocanvas.streamlit.app/](https://neurocanvas.streamlit.app/)

## Getting Started

To get started with the NeuroCanvas, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/hamdi3/NeuroCanvas.git
   ```

2. Install the required dependencies. We recommend using a virtual environment:
   ```bash
   cd NeuroCanvas
   python3.10 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Launch the web application:
   
   ```bash
   streamlit run app.py
   ```
   
4. Access the web application by opening `http://localhost:8501` in your browser.

## Contributing

Contributions are welcome and greatly appreciated! To contribute to the NeuroCanvas project, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make the desired changes and commit them:
   
   ```bash
   git commit -m "Add my feature"
   ```

4. Push to the branch:
      
   ```bash
   git push origin feature/my-feature
   ```

5. Open a pull request in the main repository.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](https://github.com/hamdi3/NeuroCanvas/blob/main/LICENSE) file for more details.

## Contact

If you have any questions, suggestions, or feedback, please feel free to contact me:

- GitHub: [github.com/hamdi3](https://github.com/hamdi3)

I'm open to collaboration and look forward to hearing from you!

---

Thank you for visiting the PRNU Predictor repository. I hope you find it useful and informative. Happy device identification using PRNU values!
