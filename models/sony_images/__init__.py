from models.sony_images import sid_bottleneck_transformer, sid_no_bottleneck, sid_original

_MODEL_DICT = {
    'sid_original': sid_original.Model,
    'sid_bottleneck_transformer_2b': sid_bottleneck_transformer.Model_2b,
    'sid_bottleneck_transformer_2b_c': sid_bottleneck_transformer.Model_2b_c,
    'sid_bottleneck_transformer_3b': sid_bottleneck_transformer.Model_3b,
    'sid_bottleneck_transformer_4b_c': sid_bottleneck_transformer.Model_4b_c,
    'sid_no_bottleneck': sid_no_bottleneck.Model
}

_INITIAL_STATE_DICT = {
    'sid_bottleneck_transformer_2b': './models/sony_images/states/sid_bottleneck_transformer_initial_2b.pt',
    'sid_bottleneck_transformer_2b_c': './models/sony_images/states/sid_bottleneck_transformer_initial_2b_c.pt',
    'sid_bottleneck_transformer_3b': './models/sony_images/states/sid_bottleneck_transformer_initial_3b.pt',
    'sid_bottleneck_transformer_4b_c': './models/sony_images/states/sid_bottleneck_transformer_initial_4b_c.pt',
}

def get_model_class(name):
    return _MODEL_DICT[name]

def get_model_initial_state(name):
    return _INITIAL_STATE_DICT[name]

