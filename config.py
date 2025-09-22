import configparser

def init_common_args(parser):
    parser.add_argument('--dataset_folder', type=str, default='./../dataset', help='location of the downloaded and unzipped dataset')
    parser.add_argument('--preprocess_folder', type=str, default='./../preprocess', help='location where preprocesses images are stored')
    parser.add_argument('--out_folder', type=str, default='./../out', help='location where logs and checkpoints are stored')
    parser.add_argument('--device_config', type=str, default='./config/cpu.ini', help='Device config to use')
    
    return parser

class DeviceConfig():
    def __init__(self, file):
        config = configparser.ConfigParser()
        config.read(file)
        self.use_cuda = config.getboolean('default', 'use_cuda')
        self.num_workers = config.getint('default', 'num_workers')
        self.train_batch_size = config.getint('default', 'train_batch_size')
        self.validation_batch_size = config.getint('default', 'validation_batch_size')
        self.auto_mixed_precision = config.getboolean('default', 'auto_mixed_precision')
        self.compile_model = config.getboolean('compile_model')
        self.preload_gts = config.getboolean('preload_gts')