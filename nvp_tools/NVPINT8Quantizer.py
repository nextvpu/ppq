from typing import Iterable, List, Set, Union

import torch
from ppq import (BaseQuantizer, 
                 QuantizationOptimizationPipeline, 
                 QuantizationOptimizationPass,
                 QuantizationSetting)
from ppq.core import (PASSIVE_OPERATIONS, ChannelwiseTensorQuantizationConfig,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.IR import BaseGraph, GraphCommandProcessor
from ppq.IR.quantize import QuantableOperation
from ppq.IR.base.graph import Operation, Variable
from ppq.IR.search import SearchableGraph
from ppq.core.quant import TensorQuantizationConfig
from ppq.api import (register_network_exporter,
                     register_network_quantizer)
from ppq.executor import BaseGraphExecutor

PASSIVE_OPERATIONS.update(
    {'GlobalAveragePool', 'AveragePool'}
)

class NvpQuantizeCrossLayerSyncPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='NVP Quantization Cross Layer Sync Pass')

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        processor = SearchableGraph(graph)

        patterns = processor.pattern_matching(
            patterns=[lambda x: x.type is not None, lambda x: x.type is not None],
            edges=[[0, 1]], exclusive=True)

        for pattern in patterns:
            pre_op, post_op = pattern
            
            if (isinstance(post_op, QuantableOperation) or
                not isinstance(pre_op, QuantableOperation)): 
                continue

            if (len(graph.get_downstream_operations(pre_op)) == 1 and 
                len(graph.get_upstream_operations(post_op)) == 1):
                pre_op.config.output_quantization_config[0].state = QuantizationStates.FP32


class NvpQuantizeGraphOutTensorPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='NVP Quantization Graph Output Tensor Pass')

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:
        processor = SearchableGraph(graph)

        for _, operation in graph.operations.items():
            if isinstance(operation, QuantableOperation) and len(graph.get_downstream_operations(operation)) == 0:
                operation.config.output_quantization_config[0].state = QuantizationStates.FP32


class NVPQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcessor]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - int(pow(2, self._num_of_bits - 1))
        self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

    def build_quant_pipeline(
        self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        quant_pipeline = super(NVPQuantizer, self).build_quant_pipeline(setting) 
        quant_pipeline.append_optimization_to_pipeline(optim_pass=NvpQuantizeCrossLayerSyncPass())

        return quant_pipeline

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = base_quant_config.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    base_quant_config.input_quantization_config[1] = \
                        ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                            convert_from = conv_weight_config,
                            offset = None, scale  = None, channel_axis = 0
                        )
                    base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = base_quant_config.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    base_quant_config.input_quantization_config[1] = \
                        ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                            convert_from = gemm_weight_config,
                            offset = None, scale  = None, channel_axis = 0
                        )
                    base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            
            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type in {'PRelu'}:
            bias_config = base_quant_config.input_quantization_config[-1]
            bias_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        
        return base_quant_config

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'MatMul', 
            'MaxPool', 'AveragePool', 'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'Sub', 'Div', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'Sigmoid', 'ReduceMean'
        }


    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {}


class NVP_INT8_Quantizer(NVPQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcessor]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)


    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.EXTENSION

register_network_quantizer(NVP_INT8_Quantizer, TargetPlatform.EXTENSION)

nvp_quant_setting = QuantizationSetting()
nvp_quant_setting.equalization = False
nvp_quant_setting.fusion_setting.fuse_conv_add = False
nvp_quant_setting.fusion_setting.fuse_activation = False
nvp_quant_setting.fusion_setting.align_elementwise_to = 'None'


