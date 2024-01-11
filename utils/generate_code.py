import re
import streamlit as st


def create_nn_code(layers):
    code_string = "import torch.nn as nn\n\n"
    code_string += "model = nn.Sequential(\n"

    for i, layer in enumerate(layers):
        layer_type = layer['Layer Type']
        params = str(re.sub(r'([a-z])\s([a-z])', r'\1_\2',
                     layer['Parameters'], flags=re.I)).replace(": ", "=")
        code_string += f"    nn.{layer_type}({params}),"
        code_string += "\n"

    code_string += ")"

    code = st.code(f"""{code_string}""", language="python")
    return code


def generate_trainer_code(components):
    counter = 0
    code_string = "import torch\nimport torchvision\n\n"

    for comp in components:
        if comp["Type"] == "Datasets":
            code_string += "# Load Dataset and create dataloader\n"
            code_string += f"train_data = torchvision.datasets.{comp['Component']}(root='data', train=True, download=True, transform=ToTensor())\n"
            code_string += f"test_data = torchvision.datasets.{comp['Component']}(root='data', train=False, download=True, transform=ToTensor())\n\n"
            code_string += f"train_dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)\n"
            code_string += f"test_dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)\n\n"
            counter += 1
        elif comp["Type"] == "Loss Function":
            code_string += "# Defining the Loss Function\n"
            component_type = comp['Component']
            params = str(re.sub(r'([a-z])\s([a-z])', r'\1_\2',
                         comp['Parameters'], flags=re.I)).replace(": ", "=")
            code_string += f"loss_fn = torch.nn.{component_type}({params})\n\n"
            counter += 1
        elif comp["Type"] == "Optimizers":
            code_string += "# Defining the Optimizer\n"
            component_type = comp['Component']
            params = str(re.sub(r'([a-z])\s([a-z])', r'\1_\2',
                         comp['Parameters'], flags=re.I)).replace(": ", "=")
            code_string += f"optim = torch.optim.{component_type}(model.parameters(),{params})\n\n"
            counter += 1

    if counter != 3:
        code_string += "# You're still messing some components have you checked if a dataset, loss function and optimizer are added?"

    elif counter == 3:
        code_string += "# Defining the training and testing functions\n"
        code_string += """def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")\n\n"""

        code_string += """def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\\n")
    """

        code_string += """
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")"""

    code = st.code(f"""{code_string}""", language="python")
    return code
