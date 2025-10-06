import configparser

def init_common_args(parser):
    parser.add_argument('--dataset_folder', type=str, default='./../dataset', help='location of the downloaded and unzipped dataset')
    parser.add_argument('--preprocess_folder', type=str, default='./../preprocess', help='location where preprocesses images are stored')
    parser.add_argument('--out_folder', type=str, default='./../out', help='location where logs and checkpoints are stored')
    parser.add_argument('--device_config', type=str, default='./config/cpu.ini', help='Device config to use')
    parser.add_argument('--model', type=str, default='sid_bottleneck_transformer_2b', help='The model identifier')
    parser.add_argument('--compile', action='store_true', default=False, help='Whether to compile the model')

    return parser

class DeviceConfig():
    def __init__(self, file):
        config = configparser.ConfigParser()
        config.read(file)
        self.use_cuda = config.getboolean('DEFAULT', 'use_cuda')
        self.num_workers = config.getint('DEFAULT', 'num_workers')
        self.train_batch_size = config.getint('DEFAULT', 'train_batch_size')
        self.validation_batch_size = config.getint('DEFAULT', 'validation_batch_size')
        self.auto_mixed_precision = config.getboolean('DEFAULT', 'auto_mixed_precision')
        self.preload_gts = config.getboolean('DEFAULT', 'preload_gts')