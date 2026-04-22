from hls4ml.model import ModelGraph
from hls4ml.utils.dependency import requires


# ----------------------Helpers---------------------
def sanitize_layer_name(layer):
    new_name = layer['name']
    if new_name[0].isdigit():
        new_name = layer['class_name'].lower() + new_name

    layer['name'] = new_name


def replace_char_inconsitency(name):
    """
    Replace some inconsistent characters that cause issues when writing into HLS.
    """
    return name.replace('.', '_')


def get_onnx_attribute(operation, name, default=None):
    from onnx import helper

    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value


def get_global_input_shape(graph, inp):
    """Return the global input shape of the graph with name inp

    Arguments:
        graph:  the onnx graph
        inp (str):  the global input name

    Returns:
        list: The shape

    Raises:
        StopIteration:  If the global input name is not found
    """
    inp_shape = next(x.type.tensor_type.shape.dim for x in graph.input if x.name == inp)
    return list(x.dim_value for x in inp_shape)


def get_input_shape(graph, node):
    """Return the input shapes of the node in the model

    Arguments:
        graph:  the onnx graph
        node:  the onnx node for which the input is desired

    Returns:
        list of lists: The shapes of all the inputs

    Raises:
        StopIteration:  If the an input name is not found in the graph
    """
    rv = []
    for inp in node.input:
        # Check for empty optional inputs (e.g., GRU) and skip
        if not inp:
            continue
        # first try regular variables
        vals = [x for x in graph.value_info if x.name == inp]
        if not vals:
            # then try outputs (possible if an output is intermediate)
            vals = [x for x in graph.output if x.name == inp]
        if not vals:
            # then try global input.
            vals = [x for x in graph.input if x.name == inp]
        if not vals:
            raise RuntimeError(f'Could not find the shape for input {inp}')
        dim = list(d.dim_value for d in vals[0].type.tensor_type.shape.dim)
        if dim:
            rv.append(dim)
    return rv


def get_tensor_shape(graph, tensor_name):
    """Return the shape of the tensor with name tensor_name

    Arguments:
        graph:  the onnx graph
        tensor_name:  the name of the tensor for which the shape is desired

    Returns:
        list: The shape of the tensor

    Raises:
        RuntimeError:  If the tensor name is not found in the graph
    """
    vals = [x for x in graph.value_info if x.name == tensor_name]
    if not vals:
        vals = [x for x in graph.output if x.name == tensor_name]
    if not vals:
        vals = [x for x in graph.input if x.name == tensor_name]
    if not vals:
        tensor = next((x for x in graph.initializer if x.name == tensor_name), None)
        if tensor is not None:
            return list(tensor.dims)
        raise RuntimeError(f'Could not find the shape for tensor {tensor_name}')

    return list(d.dim_value for d in vals[0].type.tensor_type.shape.dim)


def get_constant_value(graph, constant_name):
    tensor = next((x for x in graph.initializer if x.name == constant_name), None)
    from onnx import numpy_helper

    return numpy_helper.to_array(tensor)


def compute_pads_1d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        if layer['in_width'] % layer['stride_width'] == 0:
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)

        pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

        if auto_pad == 'SAME_UPPER':
            pads = sorted(pads)
        elif auto_pad == 'SAME_LOWER':
            pads = sorted(pads, reverse=True)
        else:  # 'VALID' padding
            pads = [0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0])

    return pads


def compute_pads_2d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        # Height
        if layer['in_height'] % layer['stride_height'] == 0:
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
        pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

        # Width
        if layer['in_width'] % layer['stride_width'] == 0:
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
        pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

        if auto_pad == 'SAME_UPPER':
            pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
        elif auto_pad == 'SAME_LOWER':
            pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
        else:  # 'VALID' padding
            pads = [0, 0, 0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])

    return pads


def _node_identifier(node):
    if node.name:
        return node.name

    return f'{node.op_type}:{"|".join(node.output)}'


class OnnxGraphContext:
    """A helper class to manage the ONNX graph and provide easier access to producers,
    consumers, initializers, etc."""

    def __init__(self, graph):
        self.graph = graph
        self.initializers = {initializer.name: initializer for initializer in graph.initializer}
        self.input_names = {inp.name for inp in graph.input}
        self.output_names = [out.name for out in graph.output]
        self.node_by_output = {}
        self.consumers = {}
        self.consumed_nodes = set()

        for node in graph.node:
            for output in node.output:
                self.node_by_output[output] = node
            for inp in node.input:
                if inp:
                    self.consumers.setdefault(inp, []).append(node)

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def is_initializer(self, name):
        return name in self.initializers

    def is_graph_input(self, name):
        return name in self.input_names

    def is_graph_output(self, name):
        return name in self.output_names

    def get_producer(self, tensor_name):
        return self.node_by_output.get(tensor_name)

    def get_consumers(self, tensor_name):
        return self.consumers.get(tensor_name, [])

    def get_single_consumer(self, tensor_name):
        consumers = self.get_consumers(tensor_name)
        if len(consumers) != 1:
            return None

        return consumers[0]

    def mark_consumed(self, node):
        self.consumed_nodes.add(_node_identifier(node))

    def is_consumed(self, node):
        return _node_identifier(node) in self.consumed_nodes


# ----------------------Layer handling---------------------
layer_handlers = {}


def register_onnx_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception(f'Layer {layer_name} already registered')
    else:
        layer_handlers[layer_name] = handler_func


def get_supported_onnx_layers():
    return list(layer_handlers.keys())


def onnx_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


def get_out_layer_names_from_layer_list(graph, layer_list):
    remaining_outputs = set(graph.output_names)
    output_layers = []

    for layer in layer_list:
        layer_outputs = set(layer.get('outputs', [layer['name']]))
        if layer_outputs & remaining_outputs:
            output_layers.append(layer['name'])
            remaining_outputs -= layer_outputs

    if remaining_outputs:
        missing_outputs = ', '.join(sorted(remaining_outputs))
        raise RuntimeError(f'Could not find the output layer for output tensor(s) {missing_outputs}')

    return output_layers


def get_constant_layers(graph, layer_list):
    referenced_constants = {output_name for output_name in graph.output_names if graph.is_initializer(output_name)}

    for layer in layer_list:
        for input_name in layer.get('inputs', []):
            if graph.is_initializer(input_name):
                referenced_constants.add(input_name)

    constant_layers = []
    for constant_name in graph.initializers:
        if constant_name not in referenced_constants:
            continue

        constant_layer = {}
        constant_layer['name'] = replace_char_inconsitency(constant_name)
        constant_layer['class_name'] = 'Constant'
        constant_layer['outputs'] = [constant_name]
        constant_layer['value'] = get_constant_value(graph, constant_name)

        sanitize_layer_name(constant_layer)
        constant_layers.append(constant_layer)

    return constant_layers


def is_onnx_recurrent_input_wrapper(node, graph):
    """Checks for the common pattern of a transpose wrapper around recurrent layers,
    (batch-first vs sequence-first).
    """
    if node.op_type != 'Transpose' or len(node.output) != 1:
        return False

    consumer = graph.get_single_consumer(node.output[0])
    if consumer is None or consumer.op_type not in ('GRU', 'LSTM', 'RNN'):
        return False

    perm = list(get_onnx_attribute(node, 'perm', []))
    return perm == [1, 0, 2]


def parse_onnx_model(onnx_model):
    """Parses the onnx model, both for configuration building and general processing.

    Args:
        onnx_model: an ONNX model object.

    Raises:
        Exception: Raised if an unsupported operation is found in the ONNX model.

    Returns:
        layer_list (list):  The onnx layers
        input_layers (list):  The input layers
        output_layers (list):  The output layers
    """
    input_layer_list = []
    parsed_layers = []
    graph = OnnxGraphContext(onnx_model.graph)

    # We don't infer the shapes because the qonnx package preprocessing does it.

    # Obtain list of input/ouput layers
    all_inputs = [x.name for x in graph.input]
    all_initializers = list(graph.initializers)
    input_layers = [x for x in all_inputs if x not in all_initializers]

    # First build the input layers
    for i, inp in enumerate(input_layers):
        input_layer = {}
        input_layer['name'] = replace_char_inconsitency(inp)
        input_layer['class_name'] = 'InputLayer'
        inp_shape = get_global_input_shape(graph, inp)
        # We only support ONNX where the first dimension is the batch dimension.
        # Remove the batch dimension in all subsequnt use
        input_layer['input_shape'] = inp_shape[1:]

        print('Input shape:', input_layer['input_shape'])
        # Clean the layer name for specific models
        sanitize_layer_name(input_layer)
        input_layers[i] = input_layer['name']

        input_layer_list.append(input_layer)

    # Defined supported layers and check for unsupported layer type
    skip_layers = ['Dropout', 'Identity']

    # Map inputs of skipped layers
    inputs_map = {}

    supported_layers = get_supported_onnx_layers() + skip_layers

    # Parse the graph operators in topological order
    print('Topology:')
    for node in graph.node:
        if graph.is_consumed(node):
            'Node has already been consumed as part of a layer handler, skipping.'
            continue
        if is_onnx_recurrent_input_wrapper(node, graph):
            'Node is a recurrent input wrapper, skipping.'
            continue

        if node.op_type not in supported_layers:
            raise Exception(f'ERROR: Unsupported operation type: {node.op_type}')

        # Note that at this point, input shape still contains batch dimension
        # in cases where it appears. That is not filtered out till later.
        input_shapes = get_input_shape(graph, node)

        if node.op_type in skip_layers:
            # Currently supported skipped layers have only one input and output
            # Skipped layers can follow each other

            # Mapping inputs
            input_name = inputs_map.get(node.input[0], node.input[0])
            output_name = node.output[0]
            inputs_map[output_name] = input_name
            continue

        input_names = [inputs_map.get(x, x) for x in node.input]

        # Process the layer
        layer = layer_handlers[node.op_type](node, input_names, input_shapes, graph)

        sanitize_layer_name(layer)
        print(f'Layer name: {layer["name"]}, layer type: {layer["class_name"]}, current shape: {input_shapes}')
        parsed_layers.append(layer)

    # Parse constant layers
    constant_layers = get_constant_layers(graph, parsed_layers)
    layer_list = input_layer_list + constant_layers + parsed_layers

    # Parse output layer(s)
    output_layers = get_out_layer_names_from_layer_list(graph, layer_list)
    print('Output layers: ', output_layers)

    return layer_list, input_layers, output_layers


@requires('onnx')
def onnx_to_hls(config):
    """Convert onnx model to hls model from configuration.

    Args:
        config (dict): ONNX configuration from yaml file or passed through API.

    Raises:
        Exception: Raised if an unsupported operation is found in the ONNX model.

    Returns:
        ModelGraph: hls4ml model object
    """

    # Extract model architecture
    print('Interpreting Model ...')

    import onnx

    onnx_model = onnx.load(config['OnnxModel']) if isinstance(config['OnnxModel'], str) else config['OnnxModel']

    layer_list, input_layers, output_layers = parse_onnx_model(onnx_model)

    #################
    # Generate HLS
    #################

    print('Creating HLS model')
    hls_model = ModelGraph.from_layer_list(config, layer_list, input_layers, output_layers)
    return hls_model
