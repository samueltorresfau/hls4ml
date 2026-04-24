import numpy as np

from hls4ml.converters.onnx_to_hls import (
    get_constant_value,
    get_onnx_attribute,
    get_tensor_shape,
    onnx_handler,
    replace_char_inconsistency,
)
from hls4ml.model.quantizers import QuantNodeQuantizer
from hls4ml.model.types import FixedPrecisionType, NamedType


def _get_initializer_data(graph, tensor_name, node, tensor_role):
    node_label = f'QONNX {node.op_type} node "{node.name}"'

    if not tensor_name:
        raise RuntimeError(f'{node_label} is missing the required {tensor_role} tensor.')

    if not graph.is_initializer(tensor_name):
        raise RuntimeError(f'{node_label} requires the {tensor_role} tensor "{tensor_name}" to be a folded initializer.')

    return get_constant_value(graph, tensor_name)


def _get_quant_node(graph, tensor_name, node, tensor_role):
    node_label = f'QONNX {node.op_type} node "{node.name}"'
    quant_node = graph.get_producer(tensor_name)

    if quant_node is None or quant_node.op_type not in ('Quant', 'IntQuant'):
        raise RuntimeError(f'{node_label} requires the {tensor_role} tensor "{tensor_name}" to be produced by a Quant node.')

    return quant_node


def _make_flat_quantizer(quant_info):
    scale = quant_info['scale']
    zeropt = quant_info['zeropt']
    bitwidth = int(np.ceil(float(quant_info['bitwidth'])))

    zeropt = np.asarray(zeropt)
    scale = np.asarray(scale)
    if zeropt.size != 1 or float(zeropt.reshape(-1)[0]) != 0.0:
        return None

    if scale.size != 1:
        return None

    scale_value = float(scale.reshape(-1)[0])
    if scale_value <= 0.0:
        return None

    mantissa, exponent = np.frexp(scale_value)
    if mantissa != 0.5:
        return None

    if quant_info['rounding_mode'] == 'ROUND':
        bn_round = 'AP_RND_CONV'
    elif quant_info['rounding_mode'] == 'FLOOR':
        bn_round = 'AP_TRN'
    else:
        return None

    bn_sat = 'AP_SAT_SYM' if quant_info['narrow'] else 'AP_SAT'
    integer = bitwidth + exponent - 1
    precision = FixedPrecisionType(bitwidth, integer, quant_info['signed'], bn_round, bn_sat)
    return QuantNodeQuantizer(precision)


def _parse_quant_info(graph, quant_node, require_initializer_input=False):
    source_name = quant_node.input[0]
    if require_initializer_input and not graph.is_initializer(source_name):
        raise RuntimeError(
            f'QONNX Quant node "{quant_node.name}" requires its source tensor "{source_name}" to be a folded initializer.'
        )

    scale = _get_initializer_data(graph, quant_node.input[1], quant_node, 'scale').astype(np.float32)
    zeropt = _get_initializer_data(graph, quant_node.input[2], quant_node, 'zero_point').astype(np.float32)
    bitwidth = _get_initializer_data(graph, quant_node.input[3], quant_node, 'bit_width').astype(np.float32)

    quant_info = {
        'scale': scale,
        'zeropt': zeropt,
        'bitwidth': float(np.asarray(bitwidth).reshape(-1)[0]),
        'narrow': bool(get_onnx_attribute(quant_node, 'narrow')),
        'signed': bool(get_onnx_attribute(quant_node, 'signed')),
        'rounding_mode': get_onnx_attribute(quant_node, 'rounding_mode'),
        'source': source_name,
    }
    quant_info['flat_quantizer'] = _make_flat_quantizer(quant_info)

    if require_initializer_input:
        quant_info['data'] = _get_initializer_data(graph, source_name, quant_node, 'source').astype(np.float32)

    return quant_info


def _get_homogeneous_quantizer(quant_infos, node, tensor_role):
    node_label = f'QONNX QuantGRUCell node "{node.name}"'
    flat_quantizers = [info['flat_quantizer'] for info in quant_infos]
    if any(quantizer is None for quantizer in flat_quantizers):
        raise RuntimeError(
            f'{node_label} only supports {tensor_role} quantization with scalar zero_point=0, scalar power-of-two '
            'scale, and ROUND/FLOOR rounding.'
        )

    reference = flat_quantizers[0]
    for quantizer in flat_quantizers[1:]:
        if quantizer.bits != reference.bits or quantizer.hls_type != reference.hls_type:
            raise RuntimeError(f'{node_label} only supports homogeneous {tensor_role} quantization across all gates in v1.')

    return reference


def _activation_quantizer_dict(name, quant_info):
    return {
        'class_name': f'quantized_{name}',
        'config': {
            'bits': int(np.ceil(quant_info['bitwidth'])),
            'integer': quant_info['flat_quantizer'].hls_type.integer - 1 if quant_info['flat_quantizer'] else None,
            'scale': np.asarray(quant_info['scale']).tolist(),
            'zeropt': np.asarray(quant_info['zeropt']).tolist(),
            'signed': quant_info['signed'],
            'narrow': quant_info['narrow'],
            'rounding_mode': quant_info['rounding_mode'],
        },
    }


def _validate_quant_gru_outputs(node, graph):
    if len(node.output) < 2 or not node.output[1]:
        return

    state_output = node.output[1]
    if graph.is_graph_output(state_output):
        raise RuntimeError(
            f'QONNX QuantGRUCell node "{node.name}" requires a trailing Unsqueeze(axis=0) when exposing the hidden state.'
        )

    consumers = graph.get_consumers(state_output)
    if len(consumers) == 0:
        return

    if len(consumers) != 1 or consumers[0].op_type != 'Unsqueeze':
        raise RuntimeError(
            f'QONNX QuantGRUCell node "{node.name}" only supports a trailing Unsqueeze(axis=0) on the hidden-state output.'
        )

    unsqueeze_node = consumers[0]
    axes = _get_initializer_data(graph, unsqueeze_node.input[1], unsqueeze_node, 'axes').reshape(-1).tolist()
    if axes != [0]:
        raise RuntimeError(
            f'QONNX QuantGRUCell node "{node.name}" only supports a trailing Unsqueeze(axis=0) on the hidden-state output.'
        )

    if not graph.is_graph_output(unsqueeze_node.output[0]):
        raise RuntimeError(
            f'QONNX QuantGRUCell node "{node.name}" only supports the hidden-state Unsqueeze as a graph output.'
        )

    graph.mark_consumed(unsqueeze_node)
    graph.ignore_output(unsqueeze_node.output[0])


def _validate_quant_gru_contract(node, graph):
    node_label = f'QONNX QuantGRUCell node "{node.name}"'

    if sum(other.op_type == 'QuantGRUCell' for other in graph.node) != 1:
        raise RuntimeError(f'{node_label} only supports a single QuantGRUCell stage in v1.')

    batch_first = get_onnx_attribute(node, 'batch_first', 0)
    if batch_first not in (0, 1):
        raise RuntimeError(f'{node_label} only supports batch_first=0 or batch_first=1 in v1.')

    if get_onnx_attribute(node, 'reverse_input', 0) != 0:
        raise RuntimeError(f'{node_label} only supports reverse_input=0 in v1.')

    initial_state_name = node.input[1]
    if graph.is_graph_input(initial_state_name):
        raise RuntimeError(f'{node_label} only supports an all-zero folded initial hidden-state initializer in v1.')

    initial_state = _get_initializer_data(graph, initial_state_name, node, 'initial hidden state')
    if not np.allclose(initial_state, 0.0):
        raise RuntimeError(f'{node_label} only supports an all-zero folded initial hidden state in v1.')

    _validate_quant_gru_outputs(node, graph)


@onnx_handler('QuantGRUCell')
def parse_quant_gru_cell(node, input_names, input_shapes, graph):
    del input_shapes

    _validate_quant_gru_contract(node, graph)
    batch_first = bool(get_onnx_attribute(node, 'batch_first', 0))

    input_quant_name = input_names[0]
    _get_quant_node(graph, input_quant_name, node, 'input')

    input_shape = get_tensor_shape(graph, input_quant_name)
    if len(input_shape) != 3:
        raise RuntimeError(
            f'QONNX QuantGRUCell node "{node.name}" expects a rank-3 sequence input tensor, got shape {input_shape}.'
        )

    gate_names = ('reset', 'update', 'new')
    weight_quant_infos = {}
    recurrent_weight_quant_infos = {}
    bias_quant_infos = {}
    recurrent_bias_quant_infos = {}

    for gate_index, gate_name in enumerate(gate_names):
        weight_quant_infos[gate_name] = _parse_quant_info(
            graph, _get_quant_node(graph, node.input[2 + gate_index], node, f'input weight {gate_name}'), True
        )
        recurrent_weight_quant_infos[gate_name] = _parse_quant_info(
            graph, _get_quant_node(graph, node.input[5 + gate_index], node, f'recurrent weight {gate_name}'), True
        )
        bias_quant_infos[gate_name] = _parse_quant_info(
            graph, _get_quant_node(graph, node.input[8 + gate_index], node, f'bias {gate_name}'), True
        )
        recurrent_bias_quant_infos[gate_name] = _parse_quant_info(
            graph, _get_quant_node(graph, node.input[11 + gate_index], node, f'recurrent bias {gate_name}'), True
        )

    output_quant_info = {
        'scale': _get_initializer_data(graph, node.input[14], node, 'output scale').astype(np.float32),
        'zeropt': _get_initializer_data(graph, node.input[15], node, 'output zero point').astype(np.float32),
        'bitwidth': float(np.asarray(_get_initializer_data(graph, node.input[16], node, 'output bit width')).reshape(-1)[0]),
        'narrow': bool(get_onnx_attribute(node, 'output_narrow')),
        'signed': bool(get_onnx_attribute(node, 'output_signed')),
        'rounding_mode': get_onnx_attribute(node, 'output_rounding_mode'),
    }
    output_quant_info['flat_quantizer'] = _make_flat_quantizer(output_quant_info)

    accumulator_names = ('reset', 'update', 'new')
    accumulator_infos = {}
    accumulator_offsets = {'reset': 17, 'update': 20, 'new': 23}
    for acc_name, offset in accumulator_offsets.items():
        accumulator_infos[acc_name] = {
            'scale': _get_initializer_data(graph, node.input[offset], node, f'{acc_name} accumulator scale').astype(
                np.float32
            ),
            'zeropt': _get_initializer_data(
                graph, node.input[offset + 1], node, f'{acc_name} accumulator zero point'
            ).astype(np.float32),
            'bitwidth': float(
                np.asarray(
                    _get_initializer_data(graph, node.input[offset + 2], node, f'{acc_name} accumulator bit width')
                ).reshape(-1)[0]
            ),
            'narrow': bool(get_onnx_attribute(node, f'{acc_name}_acc_narrow')),
            'signed': bool(get_onnx_attribute(node, f'{acc_name}_acc_signed')),
            'rounding_mode': get_onnx_attribute(node, f'{acc_name}_acc_rounding_mode'),
        }
        accumulator_infos[acc_name]['flat_quantizer'] = _make_flat_quantizer(accumulator_infos[acc_name])

    activation_infos = {}
    for act_name, offset in (('sigmoid', 26), ('tanh', 29)):
        activation_infos[act_name] = {
            'scale': _get_initializer_data(graph, node.input[offset], node, f'{act_name} scale').astype(np.float32),
            'zeropt': _get_initializer_data(graph, node.input[offset + 1], node, f'{act_name} zero point').astype(
                np.float32
            ),
            'bitwidth': float(
                np.asarray(_get_initializer_data(graph, node.input[offset + 2], node, f'{act_name} bit width')).reshape(-1)[
                    0
                ]
            ),
            'narrow': bool(get_onnx_attribute(node, f'{act_name}_narrow')),
            'signed': bool(get_onnx_attribute(node, f'{act_name}_signed')),
            'rounding_mode': get_onnx_attribute(node, f'{act_name}_rounding_mode'),
        }
        activation_infos[act_name]['flat_quantizer'] = _make_flat_quantizer(activation_infos[act_name])

    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'GRU'
    layer['inputs'] = [input_quant_name]
    layer['outputs'] = [node.output[0]]
    layer['activation'] = 'tanh'
    layer['recurrent_activation'] = 'sigmoid'
    layer['return_sequences'] = True
    layer['return_state'] = False
    layer['direction'] = 'forward'
    layer['time_major'] = not batch_first
    layer['apply_reset_gate'] = 'before'
    layer['pytorch'] = True
    layer['pass_initial_states'] = False
    layer['n_timesteps'] = input_shape[1] if batch_first else input_shape[0]
    layer['n_in'] = input_shape[2]
    layer['n_out'] = int(np.asarray(weight_quant_infos['reset']['data']).shape[0])

    layer['weight_data'] = np.concatenate(
        [weight_quant_infos[gate_name]['data'].transpose() for gate_name in gate_names], axis=1
    )
    layer['recurrent_weight_data'] = np.concatenate(
        [recurrent_weight_quant_infos[gate_name]['data'].transpose() for gate_name in gate_names], axis=1
    )
    layer['bias_data'] = np.concatenate([bias_quant_infos[gate_name]['data'] for gate_name in gate_names], axis=0)
    layer['recurrent_bias_data'] = np.concatenate(
        [recurrent_bias_quant_infos[gate_name]['data'] for gate_name in gate_names], axis=0
    )

    weight_quantizer = _get_homogeneous_quantizer(
        [weight_quant_infos[gate_name] for gate_name in gate_names], node, 'input-weight'
    )
    recurrent_weight_quantizer = _get_homogeneous_quantizer(
        [recurrent_weight_quant_infos[gate_name] for gate_name in gate_names],
        node,
        'recurrent-weight',
    )
    bias_quantizer = _get_homogeneous_quantizer(
        [bias_quant_infos[gate_name] for gate_name in gate_names]
        + [recurrent_bias_quant_infos[gate_name] for gate_name in gate_names],
        node,
        'bias',
    )
    accum_quantizer = _get_homogeneous_quantizer(
        [accumulator_infos[acc_name] for acc_name in accumulator_names], node, 'accumulator'
    )

    layer['weight_quantizer'] = weight_quantizer
    layer['recurrent_weight_quantizer'] = recurrent_weight_quantizer
    layer['bias_quantizer'] = bias_quantizer
    layer['accum_quantizer'] = accum_quantizer
    layer['accum_t'] = NamedType(name=f'{replace_char_inconsistency(node.name)}_accum_t', precision=accum_quantizer.hls_type)

    if output_quant_info['flat_quantizer'] is not None:
        layer['result_t'] = NamedType(
            name=f'{replace_char_inconsistency(node.name)}_result_t',
            precision=output_quant_info['flat_quantizer'].hls_type,
        )

    layer['activation_quantizer'] = _activation_quantizer_dict('tanh', activation_infos['tanh'])
    layer['recurrent_activation_config'] = {
        'class_name': 'Activation',
        'recurrent_activation': 'sigmoid',
        'activation_quantizer': _activation_quantizer_dict('sigmoid', activation_infos['sigmoid']),
    }

    return layer
