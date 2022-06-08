from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)


def create_quantization_preprocessor():
    # Create a quantization preprocessor to determine the nodes to exclude
    quantization_preprocessor = QuantizationPreprocessor()

    # Exclude the nodes constituting LayerNorm
    quantization_preprocessor.register_pass(ExcludeLayerNormNodes())
    # Exclude the nodes constituting GELU
    quantization_preprocessor.register_pass(ExcludeGeLUNodes())
    # Exclude the residual connection Add nodes
    quantization_preprocessor.register_pass(ExcludeNodeAfter("Add", "Add"))
    # Exclude the Add nodes following the Gather operator
    quantization_preprocessor.register_pass(ExcludeNodeAfter("Gather", "Add"))
    # Exclude the Add nodes followed by the Softmax operator
    quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Add", "Softmax"))

    return quantization_preprocessor
