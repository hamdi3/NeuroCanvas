import re
import streamlit as st
import io
import sys
import traceback


def create_nn_code(layers):
    code_string = "import torch.nn as nn\n\n"
    code_string += "model = nn.Sequential(\n"

    for i, layer in enumerate(layers):
        layer_type = layer['Layer Type']
        params = str(re.sub(r'([a-z])\s([a-z])', r'\1_\2',
                     layer['Parameters'], flags=re.I)).replace(": ", "=")
        code_string += f"    nn.{layer_type}({params}),"
        code_string += "\n"

    code_string += ")\n"

    return code_string


def generate_trainer_code(components):
    counter = 0
    complete = False
    code_string = "import torch\n"
    code_string += "import torchvision\n\n"
    code_string += "device = 'cuda' if torch.backends.cuda.is_built() else 'cpu'\n\n"

    for comp in components:
        if comp["Type"] == "Datasets":
            code_string += "# Load Dataset and create dataloader\n"
            code_string += f"train_data = torchvision.datasets.{comp['Component']}(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())\n"
            code_string += f"test_data = torchvision.datasets.{comp['Component']}(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())\n\n"
            code_string += f"train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)\n"
            code_string += f"test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)\n\n"
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
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.view(X.shape[0], -1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    train_loss /= num_batches
    print(f"Train Error:\\n Avg loss {loss:>7f}\\n")\n\n"""

        code_string += """def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(X.shape[0], -1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\\n")
    """

        code_string += """
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\\n-------------------------------")
    train(train_dataloader, model, loss_fn, optim)
    test(test_dataloader, model, loss_fn)

# Saving the model
scripted_model = torch.jit.script(model)  # Convert the model to TorchScript
scripted_model.save('model_scripted.pt')  # Save the scripted model
print("Model was saved successfully")
"""

        complete = True

    return code_string, complete


def execute_code():
    model_code = create_nn_code(st.session_state["layers"])
    trainer_code, complete = generate_trainer_code(st.session_state["trainer"])

    if complete:
        code_string = model_code + trainer_code
        # Create a temporary redirect for stdout and stderr
        temp_stdout = io.StringIO()
        temp_stderr = io.StringIO()
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr
        try:
            with st.spinner('Training the model...'):
                exec(code_string, globals())  # Execute the code
        except Exception:
            st.error('An error occurred:\n' + traceback.format_exc())
        finally:
            sys.stdout = sys.__stdout__  # Reset stdout to normal
            sys.stderr = sys.__stderr__  # Reset stderr to normal
        output = temp_stdout.getvalue()  # Get the stdout string
        errors = temp_stderr.getvalue()  # Get the stderr string
        if output:
            st.code(output)  # Write the output to Streamlit
        if errors:
            st.markdown('**Errors:**')
            st.code(errors)  # Write the errors to Streamlit

        temp_stdout.close()
        temp_stderr.close()

    else:
        st.warning("Your code is still missing something", icon="⚠️")
