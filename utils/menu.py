import streamlit as st

def delete_layer():
    # Delete the last added layer
    if st.session_state["layers"] and st.session_state["G"]:
        last_node = len(st.session_state.layers) - 1
        st.session_state.G.remove_node(last_node)
        st.session_state.layers.pop()
    st.toast('layer deleted', icon="⛔")

def add_layer(layer_type, layer_params):
    # Add the configured layer to the network
    st.session_state.layers.append({
        'Layer Type': layer_type,
        'Parameters': ', '.join(f'{key.replace("_"," ")}: {value}' for key, value in layer_params.items())
    })

    # Add a node to the graph for the new layer
    st.session_state.G.add_node(
        len(st.session_state.layers) - 1, label=layer_type)

    # If this is not the first layer, add an edge from the previous layer
    if len(st.session_state.layers) > 1:
        st.session_state.G.add_edge(
            len(st.session_state.layers) - 2, len(st.session_state.layers) - 1)

    st.toast('layer added', icon="✅")

def render_model_menu():
    # Define the categories
    categories = {
        'Linear Layers': ['Linear', 'Bilinear'],
        'Convolution Layers': ['Conv2D', 'ConvTranspose2d'],
        'Pooling layers': ['MaxPool2d', 'AvgPool2d'],
        'Normalization Layers': ['BatchNorm2d', 'GroupNorm'],
        'Dropout Layers': ['Dropout'],
        'Sparse Layers': ['Embedding'],
        'Activations': ['ReLU', 'Sigmoid', 'Softmax'],
    }

    # Initialize the layer parameters dictionary
    layer_params = {}

    # Create a dropdown for the categories
    category = st.sidebar.selectbox(
        'Select the category', list(categories.keys()))

    # Create a dropdown for the layers in the selected category
    layer_type = st.sidebar.selectbox(
        f'Select the layer type ({category})', categories[category])

    # Configure the parameters based on the selected layer type
    ############################## Linear Layers ###########################
    if layer_type == 'Linear':
        layer_params['in_features'] = st.sidebar.number_input(
            'Input Size', value=0, help="size of each input sample")
        layer_params['out_features'] = st.sidebar.number_input(
            'Output Size', value=0, help="size of each output sample")
        layer_params['bias'] = st.sidebar.checkbox(
            'Bias', value=False, help="adds a learnable bias to the output")

    elif layer_type == 'Bilinear':
        layer_params['in1_features'] = st.sidebar.number_input(
            'First Input Size', value=0, help="size of each first input sample")
        layer_params['in2_features'] = st.sidebar.number_input(
            'Second Input Size', value=0, help="size of each second input sample")
        layer_params['out_features'] = st.sidebar.number_input(
            'Output Size', value=0, help="size of each output sample")
        layer_params['bias'] = st.sidebar.checkbox(
            'Bias', value=False, help="adds a learnable bias to the output.")

    ############################## Convolution Layers #######################
    elif layer_type == 'Conv2D':
        layer_params['in_channels'] = st.sidebar.number_input(
            'Input Size', value=0, help="Number of input features")
        layer_params['out_channels'] = st.sidebar.number_input(
            'Output size', value=0, help="Number of output features")
        layer_params['kernel_size'] = st.sidebar.number_input(
            'Kernel size', value=0, help="Size of the kernel")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['stride'] = st.sidebar.number_input(
                'Stride', value=1, help="controls the stride for the cross-correlation")
            layer_params['padding'] = st.sidebar.number_input(
                'Padding', value=0, help="controls the amount of implicit zero padding on both sides")
            layer_params['output_padding'] = st.sidebar.number_input(
                'Output Padding', value=0, help="controls the additional size added to one side of the output shape")
            layer_params['dilation'] = st.sidebar.number_input(
                'Dilation', value=1, help="controls the spacing between the kernel points")
            layer_params['groups'] = st.sidebar.number_input(
                'Groups', value=1, help="controls the connections between inputs and outputs")
            layer_params['bias'] = st.sidebar.checkbox(
                'Bias', value=True, help="adds a learnable bias to the output")

    elif layer_type == 'ConvTranspose2d':
        layer_params['in_channels'] = st.sidebar.number_input(
            'Input Size', value=0, help="Number of input features")
        layer_params['out_channels'] = st.sidebar.number_input(
            'Output size', value=0, help="Number of output features")
        layer_params['kernel_size'] = st.sidebar.number_input(
            'Kernel size', value=0, help="Size of the kernel")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['stride'] = st.sidebar.number_input(
                'Stride', value=1, help="controls the stride for the cross-correlation")
            layer_params['padding'] = st.sidebar.number_input(
                'Padding', value=0, help="controls the amount of padding applied to the input")
            layer_params['dilation'] = st.sidebar.number_input(
                'Dilation', value=1, help="controls the spacing between the kernel points")
            layer_params['groups'] = st.sidebar.number_input(
                'Groups', value=1, help="controls the connections between inputs and outputs")
            layer_params['bias'] = st.sidebar.checkbox(
                'Bias', value=True, help="adds a learnable bias to the output")

    ############################## Pooling Layers #############################
    elif layer_type == 'MaxPool2d':
        layer_params['kernel_size'] = st.sidebar.number_input(
            'Kernel size', value=0, help="Size of the kernel")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['stride'] = st.sidebar.number_input(
                'Stride', value=0, help="controls the stride for the cross-correlation")
            layer_params['padding'] = st.sidebar.number_input(
                'Padding', value=0, help="controls the amount of padding applied to the input")
            layer_params['dilation'] = st.sidebar.number_input(
                'Dilation', value=1, help="controls the spacing between the kernel points")
            layer_params['return_indices'] = st.sidebar.checkbox(
                'Return Indices', value=False, help=" if True, will return the max indices along with the outputs")
            layer_params['ceil_mode'] = st.sidebar.checkbox(
                'Ceil Mode', value=False, help="when True, will use ceil instead of floor to compute the output shape")

    elif layer_type == 'AvgPool2d':
        layer_params['kernel_size'] = st.sidebar.number_input(
            'Kernel size', value=0, help="Size of the kernel")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['stride'] = st.sidebar.number_input(
                'Stride', value=0, help="controls the stride for the cross-correlation")
            layer_params['padding'] = st.sidebar.number_input(
                'Padding', value=0, help="controls the amount of padding applied to the input")
            layer_params['ceil_mode'] = st.sidebar.checkbox(
                'Ceil Mode', value=False, help="when True, will use ceil instead of floor to compute the output shape")
            layer_params['count_include_pad'] = st.sidebar.checkbox(
                'Include Pad in Count', value=False, help="when True, will include the zero-padding in the averaging calculation")

    ############################## Normalization Layers #######################
    elif layer_type == 'BatchNorm2d':
        layer_params['num_features'] = st.sidebar.number_input(
            'Number of Features', value=0, help="C from an expected input of size (N,C,H,W)")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['eps'] = st.sidebar.number_input(
                'Eps', value=1e-5, step=1e-5,format="%.5e", help="a value added to the denominator for numerical stability")
            layer_params['momentum'] = st.sidebar.number_input(
                'Momentum', value=0.1, step=0.1, help="the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average)")
            layer_params['affine'] = st.sidebar.checkbox(
                'Affine', value=True, help="when set to True, this module has learnable affine parameters")
            layer_params['track_running_stats'] = st.sidebar.checkbox(
                'Track Running Stats', value=True, help="when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes")

    elif layer_type == 'GroupNorm':
        layer_params['num_groups'] = st.sidebar.number_input(
            'Number of Groups', value=0, help="number of groups to separate the channels into")
        layer_params['num_channels'] = st.sidebar.number_input(
            'Number of Channels', value=0, help="number of channels expected in input")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['eps'] = st.sidebar.number_input(
                'Eps', value=1e-5, step=1e-5, format="%.5e", help="a value added to the denominator for numerical stability")
            layer_params['affine'] = st.sidebar.checkbox(
                'Affine', value=True, help="when set to True, this module has learnable affine parameters")

    ############################## Dropout Layers ##############################
    elif layer_type == 'Dropout':
        layer_params['p'] = st.sidebar.number_input(
            'Probability', value=0.5, step=0.1, help="probability of an element to be zeroed")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['inplace'] = st.sidebar.checkbox(
                'Inplace', value=False, help="If set to True, will do this operation in-place")

    ############################## Sparse Layers ##############################
    elif layer_type == 'Embedding':
        layer_params['num_embeddings'] = st.sidebar.number_input(
            'Number of Embeddings', value=0, help="size of the dictionary of embeddings")
        layer_params['embedding_dim'] = st.sidebar.number_input(
            'Embedding Dimension', value=0, help="the size of each embedding vector")
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            layer_params['padding_idx'] = st.sidebar.number_input(
                'Padding Index', value=None, help="If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros, but can be updated to another value to be used as the padding vector")
            layer_params['max_norm'] = st.sidebar.number_input(
                'Max Normalization', value=None, help="If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm")
            layer_params['norm_type'] = st.sidebar.number_input(
                'Type of Normalization', value=2.0, help="The p of the p-norm to compute for the max_norm option")
            layer_params['scale_grad_by_freq'] = st.sidebar.checkbox(
                'Scale by Frequency', value=False, help="this will scale gradients by the inverse of frequency of the words in the mini-batch")
            layer_params['sparse'] = st.sidebar.checkbox(
                'Sparce', value=False, help="If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients")

    ############################# Activations #####################################
    elif layer_type == 'ReLU':
        layer_params['inplace'] = st.sidebar.checkbox(
            'Inplace', value=False, help="do the operation in-place")

    elif layer_type == 'Softmax':
        layer_params['dim'] = st.sidebar.number_input(
            'dimension', value=0, help="A dimension along which Softmax will be computed (so every slice along dim will sum to 1)")

    ############################## Buttons ###########################################
    # Create two columns to add the buttons next to one another
    col1, col2 = st.sidebar.columns(2)

    # Add the layer when the button in the first column is clicked
    if col1.button(f'Add {layer_type} layer'):
        add_layer(layer_type, layer_params)

    # Delete the last layer when the button in the second column is clicked
    if col2.button('Delete last layer', type="primary"):
        delete_layer()

def delete_from_trainer():
    # Delete the last added layer
    if st.session_state.trainer:
        st.session_state.trainer.pop()
    st.toast('Trainer Component deleted', icon="⛔")

def add_to_trainer(option, component, params):
    # Check if the component has already been added
    if any(option == comp["Type"] for comp in st.session_state.trainer):
        st.toast(f'You already have that component', icon="❌")
    else:
        # Add the configured layer to the network
        st.session_state.trainer.append({
            'Type': option,
            'Component': component,
            'Parameters': ', '.join(f'{key.replace("_"," ")}: {value}' for key, value in params.items())
        })

        st.toast('Component added to trainer', icon="✅")


def render_train_menu():
    # Define the training options
    training_options = {
        'Datasets': ['MNIST', 'FashionMNIST'],
        'Loss Function': ['MSELoss', 'CrossEntropyLoss'],
        'Optimizers': ['Adagrad', 'Adam'],
    }

    # Initialize the trainer parameters dictionary
    component_params = {}
    
    # Create a dropdown for the training options
    option = st.sidebar.selectbox(
        'Select the training option', list(training_options.keys()))

    # Create a dropdown for the specific training option
    component = st.sidebar.selectbox(
        f'Select the specific {option}', training_options[option])
    
    ############################## Loss Functions ###################################
    if component == 'MSELoss':
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            component_params['size_average'] = st.sidebar.checkbox(
                'Average the Losses', value=True, help="By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch")
            component_params['reduce'] = st.sidebar.checkbox(
                'Reduce', value=True, help="By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average")
            component_params['reduction'] = st.sidebar.selectbox('Select Type of Reduction', ['mean' , 'sum', 'none'],
                help="Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction")
    
    elif component == 'CrossEntropyLoss':
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            component_params['size_average'] = st.sidebar.checkbox(
                'Average the Losses', value=True, help="By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch")
            component_params['ignore_index'] = st.sidebar.number_input(
                'Index to Ignore', value=-100, help="Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices")
            component_params['reduce'] = st.sidebar.checkbox(
                'Reduce', value=True, help="By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average")
            component_params['reduction'] = st.sidebar.selectbox('Select Type of Reduction', ['mean' , 'sum', 'none'],
                help="Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction")
            component_params['label_smoothing'] = st.sidebar.number_input(
                'Label Smooting Value', value=0.0, min_value=0.0, max_value=1.0, step=0.1, help="Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision")

    ############################## Optimizers ########################################
    elif component == 'Adagrad':
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            component_params['lr'] = st.sidebar.number_input(
                'Learning Rate', value=1e-2, step=1e-2, format="%.2e")
            component_params['lr_decay'] = st.sidebar.number_input(
                'Learning Rate Decay', value=0.0, step=1e-1, format="%.1e")
            component_params['weight_decay'] = st.sidebar.number_input(
                'Weight Decay', value=0.0, step=1e-1, format="%.1e", help="This uses L2 penalty")
            component_params['eps'] = st.sidebar.number_input(
                'Epsilon', value=1e-10, step=1e-10,format="%.10e", help="term added to the denominator to improve numerical stability")
            component_params['foreach'] = st.sidebar.checkbox(
                'For Each Cuda Device', value=None, help="whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more performant. Note that the foreach implementation uses ~ sizeof(params) more peak memory than the for-loop version due to the intermediates being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer parameters through the optimizer at a time or switch this flag to False")
            component_params['maximize'] = st.sidebar.checkbox(
                'Maximize Parameters', value=False, help="maximize the params based on the objective, instead of minimizing")
            component_params['differentiable'] = st.sidebar.checkbox(
                'Differentiable', value=False, help="whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you dont intend to run autograd through this instance")
    
    elif component == 'Adam':
        on = st.sidebar.toggle('Show optional parameters?')
        if on:
            component_params['lr'] = st.sidebar.number_input(
                'Learning Rate', value=1e-2, step=1e-2, format="%.2e")
            component_params['eps'] = st.sidebar.number_input(
                'Epsilon', value=1e-10, step=1e-10, format="%.10e", help="term added to the denominator to improve numerical stability")
            component_params['weight_decay'] = st.sidebar.number_input(
                'Weight Decay', value=0.0, step=1e-1, format="%.1e", help="This uses L2 penalty")            
            component_params['amsgrad'] = st.sidebar.checkbox(
                'Use AMSGrad variant', value=False, help="whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond")
            component_params['foreach'] = st.sidebar.checkbox(
                'For Each Cuda Device', value=None, help="whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more performant. Note that the foreach implementation uses ~ sizeof(params) more peak memory than the for-loop version due to the intermediates being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer parameters through the optimizer at a time or switch this flag to False")
            component_params['maximize'] = st.sidebar.checkbox(
                'Maximize Parameters', value=False, help="maximize the params based on the objective, instead of minimizing")
            component_params['capturable'] = st.sidebar.checkbox(
                'Capture in a CUDA Graph', value=False, help="whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed performance, so if you dont intend to graph capture this instance, leave it")
            component_params['differentiable'] = st.sidebar.checkbox(
                'Differentiable', value=False, help="whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you dont intend to run autograd through this instance")

    ############################## Buttons ###########################################
    # Create two columns to add the buttons next to one another
    col1, col2 = st.sidebar.columns(2)

    # Add the layer when the button in the first column is clicked
    if col1.button(f'Add {component} component'):
        add_to_trainer(option, component, component_params)

    # Delete the last layer when the button in the second column is clicked
    if col2.button('Delete last component', type="primary"):
        delete_from_trainer()


def render_menu():
    # Create a switcher for the menus
    menu = st.sidebar.radio('Menu', ['Model', 'Trainer'])
    if menu == 'Model':
        render_model_menu()
    else:
        render_train_menu()