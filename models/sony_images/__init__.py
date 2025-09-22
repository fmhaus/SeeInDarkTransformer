import sid_bottleneck_transformer, sid_no_bottleneck, sid_original

_MODEL_DICT = {
    'sid_original': sid_original.Model,
    'sid_bottleneck_transformer_2b': sid_bottleneck_transformer.Model_2b,
    'sid_bottleneck_transformer_3b': sid_bottleneck_transformer.Model_3b,
    'sid_bottleneck_transformer_4b_c': sid_bottleneck_transformer.Model_4b_c,
    'sid_no_bottleneck': sid_no_bottleneck.Model
}

def get_model_class(name):
    return _MODEL_DICT[name]