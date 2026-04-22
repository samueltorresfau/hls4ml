import numpy as np

from hls4ml.converters.onnx_to_hls import get_constant_value, get_onnx_attribute, get_tensor_shape, onnx_handler


def _get_initializer_data(graph, tensor_name, node, tensor_role, required=True):
    node_label = f'ONNX {node.op_type} node "{node.name}"'

    if not tensor_name:
        if required:
            raise RuntimeError(f'{node_label} is missing the required {tensor_role} tensor.')
        return None

    if not graph.is_initializer(tensor_name):
        raise RuntimeError(
            f'{node_label} requires the {tensor_role} tensor "{tensor_name}" to be folded to an initializer before '
            'conversion.'
        )

    return get_constant_value(graph, tensor_name)


def _get_squeeze_axes(node, graph):
    if len(node.input) > 1 and node.input[1]:
        axes = get_constant_value(graph, node.input[1])
    else:
        axes = get_onnx_attribute(node, 'axes', [])

    return [int(axis) for axis in np.array(axes).reshape(-1)]


def _parse_recurrent_wrapper(node, graph):
    """Parse the expected input and output wrapper pattern for ONNX recurrent nodes in the ONNX/QONNX frontend,
    which is transpose(input) -> GRU/LSTM/RNN -> Squeeze -> Transpose (pattern added by QONNX cleanup_model function)"""
    node_label = f'ONNX {node.op_type} node "{node.name}"'

    input_transpose = graph.get_producer(node.input[0])
    if input_transpose is None or input_transpose.op_type != 'Transpose' or graph.get_single_consumer(node.input[0]) != node:
        raise RuntimeError(f'{node_label} requires a cleaned batch-first input transpose with perm=[1, 0, 2].')

    perm = list(get_onnx_attribute(input_transpose, 'perm', []))
    if perm != [1, 0, 2]:
        raise RuntimeError(f'{node_label} expects the cleaned input transpose to use perm=[1, 0, 2], got perm={perm}.')

    squeeze_node = graph.get_single_consumer(node.output[0])
    if squeeze_node is None or squeeze_node.op_type != 'Squeeze':
        raise RuntimeError(f'{node_label} requires a cleaned batch-first output wrapper of Squeeze -> Transpose.')

    axes = _get_squeeze_axes(squeeze_node, graph)
    if axes != [1]:
        raise RuntimeError(f'{node_label} expects the cleaned Squeeze wrapper to remove axis 1, got axes={axes}.')

    output_transpose = graph.get_single_consumer(squeeze_node.output[0])
    if output_transpose is None or output_transpose.op_type != 'Transpose':
        raise RuntimeError(f'{node_label} requires a cleaned batch-first output wrapper of Squeeze -> Transpose.')

    perm = list(get_onnx_attribute(output_transpose, 'perm', []))
    if perm != [1, 0, 2]:
        raise RuntimeError(f'{node_label} expects the cleaned output transpose to use perm=[1, 0, 2], got perm={perm}.')

    # Mark the wrapper nodes as consumed so they are ignored during conversion.
    graph.mark_consumed(input_transpose)
    graph.mark_consumed(squeeze_node)
    graph.mark_consumed(output_transpose)

    return input_transpose.input[0], output_transpose.output[0]


def _get_initial_state_input(graph, tensor_name, node, state_name):
    node_label = f'ONNX {node.op_type} node "{node.name}"'

    if not tensor_name:
        return None

    if graph.is_graph_input(tensor_name):
        return tensor_name

    if not graph.is_initializer(tensor_name):
        raise RuntimeError(f'{node_label} only supports {state_name} as a graph input or a folded zero initializer.')

    initial_state = get_constant_value(graph, tensor_name)
    if not np.allclose(initial_state, 0.0):
        # Note: Non-zero initial states might be supported (initializer layer plus Add layer)
        raise RuntimeError(f'{node_label} only supports zero-valued folded {state_name} initializers in v1.')

    return None


# GRU layer handler
def _check_gru_attributes(node, graph):
    "Perform checks on the GRU node attributes to ensure these are compatible with the current ONNX frontend support"
    node_label = f'ONNX GRU node "{node.name}"'

    if get_onnx_attribute(node, 'direction', 'forward') != 'forward':
        raise RuntimeError(f'{node_label} only supports direction="forward" in the ONNX/QONNX frontend v1.')

    if get_onnx_attribute(node, 'layout', 0) != 0:
        raise RuntimeError(f'{node_label} only supports layout=0 in the ONNX/QONNX frontend v1.')

    if get_onnx_attribute(node, 'linear_before_reset', 0) != 1:
        raise RuntimeError(f'{node_label} only supports linear_before_reset=1 in the ONNX/QONNX frontend v1.')

    if get_onnx_attribute(node, 'clip') is not None:
        raise RuntimeError(f'{node_label} does not support the clip attribute.')

    activations = get_onnx_attribute(node, 'activations')
    if activations is not None:
        activations = [value.decode() if isinstance(value, bytes) else value for value in activations]
        if activations != ['Sigmoid', 'Tanh']:
            raise RuntimeError(f'{node_label} only supports the default activations ["Sigmoid", "Tanh"], got {activations}.')

    for attr_name in ('activation_alpha', 'activation_beta'):
        values = get_onnx_attribute(node, attr_name)
        if values is not None and len(values) != 0:
            raise RuntimeError(f'{node_label} does not support custom {attr_name} values.')

    if len(node.input) > 4 and node.input[4]:
        raise RuntimeError(f'{node_label} does not support the sequence_lens input.')

    if len(node.output) > 1 and node.output[1]:
        hidden_state_output = node.output[1]
        if graph.is_graph_output(hidden_state_output) or graph.get_consumers(hidden_state_output):
            raise RuntimeError(f'{node_label} does not support exporting or consuming the hidden state output Y_h.')


@onnx_handler('GRU')
def parse_gru_layer(node, input_names, input_shapes, graph):
    del input_names, input_shapes

    _check_gru_attributes(node, graph)

    input_name, output_name = _parse_recurrent_wrapper(node, graph)
    input_shape = get_tensor_shape(graph, input_name)
    if len(input_shape) != 3:
        raise RuntimeError(
            f'ONNX GRU node "{node.name}" expects a batch-first input tensor with rank 3, got shape {input_shape}.'
        )

    weight_data = _get_initializer_data(graph, node.input[1], node, 'W').astype(np.float32)
    recurrent_weight_data = _get_initializer_data(graph, node.input[2], node, 'R').astype(np.float32)

    if weight_data.ndim != 3 or weight_data.shape[0] != 1:
        raise RuntimeError(
            f'ONNX GRU node "{node.name}" expects W with shape [1, 3 * hidden_size, input_size], got {weight_data.shape}.'
        )

    if recurrent_weight_data.ndim != 3 or recurrent_weight_data.shape[0] != 1:
        raise RuntimeError(
            f'ONNX GRU node "{node.name}" expects R with shape [1, 3 * hidden_size, hidden_size], got '
            f'{recurrent_weight_data.shape}.'
        )

    hidden_size = get_onnx_attribute(node, 'hidden_size')
    if hidden_size is None:
        hidden_size = weight_data.shape[1] // 3

    if weight_data.shape[1] != hidden_size * 3:
        raise RuntimeError(
            f'ONNX GRU node "{node.name}" has inconsistent hidden_size={hidden_size} for W shape {weight_data.shape}.'
        )

    if recurrent_weight_data.shape[1] != hidden_size * 3 or recurrent_weight_data.shape[2] != hidden_size:
        raise RuntimeError(
            f'ONNX GRU node "{node.name}" has inconsistent hidden_size={hidden_size} for R shape '
            f'{recurrent_weight_data.shape}.'
        )

    bias_tensor = _get_initializer_data(graph, node.input[3] if len(node.input) > 3 else '', node, 'B', required=False)
    if bias_tensor is None:
        bias_data = np.zeros(hidden_size * 3, dtype=np.float32)
        recurrent_bias_data = np.zeros(hidden_size * 3, dtype=np.float32)
    else:
        bias_tensor = bias_tensor.astype(np.float32)
        if bias_tensor.shape != (1, hidden_size * 6):
            raise RuntimeError(
                f'ONNX GRU node "{node.name}" expects B with shape [1, 6 * hidden_size], got {bias_tensor.shape}.'
            )

        bias_data = bias_tensor[0, : hidden_size * 3]
        recurrent_bias_data = bias_tensor[0, hidden_size * 3 :]

    layer_inputs = [input_name]
    initial_state_input = _get_initial_state_input(graph, node.input[5] if len(node.input) > 5 else '', node, 'initial_h')
    if initial_state_input is not None:
        layer_inputs.append(initial_state_input)

    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'GRU'
    layer['inputs'] = layer_inputs
    layer['outputs'] = [output_name]
    layer['activation'] = 'tanh'
    layer['recurrent_activation'] = 'sigmoid'
    layer['return_sequences'] = True
    layer['return_state'] = False
    layer['direction'] = 'forward'
    layer['time_major'] = False
    layer['apply_reset_gate'] = 'after'
    layer['pytorch'] = False
    layer['pass_initial_states'] = initial_state_input is not None
    layer['n_timesteps'] = input_shape[1]
    layer['n_in'] = input_shape[2]
    layer['n_out'] = hidden_size
    layer['weight_data'] = weight_data[0].transpose()
    layer['recurrent_weight_data'] = recurrent_weight_data[0].transpose()
    layer['bias_data'] = bias_data
    layer['recurrent_bias_data'] = recurrent_bias_data

    return layer
