import re

def create_nn_code(layers):
    code_string = "import torch.nn as nn\n\n"
    code_string += "model = nn.Sequential(\n"

    for i, layer in enumerate(layers):
        layer_type = layer['Layer Type']
        params = str(re.sub(r'([a-z])\s([a-z])', r'\1_\2', layer['Parameters'], flags=re.I)).replace(": ", "=")
        code_string += f"    nn.{layer_type}({params}),"
        code_string += "\n"

    code_string += ")"
    return code_string

