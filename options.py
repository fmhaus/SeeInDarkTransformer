class Options():
    def __init__(self):
        pass
    
    def init(self, parser):
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='number of images processed simultaneously')
        parser.add_argument('--effective_batch_size', type=int, default=24, help='number of images processed before updating weights')
        parser.add_argument('--validation_batch_size', type=int, default=1, help='number of images processed simultaneously for validation')
        parser.add_argument('--resume_epoch', type=int, default=0, help='epoch to resume training from (0 = train from zero)')
        parser.add_argument('--total_epochs', type=int, default=200, help='toal number of epochs')
        parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs (0 = no warmup)')
        parser.add_argument('--auto_mixed_precision', action='store_true', default=False, help='whether to use auto mixed precision (AMP)')
        parser.add_argument('--augment_images_epoch', type=int, default=5, help='After what epoch images should be augmented (with random crops and flips)')
        parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load the optimizer (and lr schedule) when using resume or not')
        parser.add_argument('--preload_gts', action='store_true', default=False, help='Keeps the GTs in memory')
        parser.add_argument('--compile_model', action='store_true', default=False, help='Whether to compile to the model to speedup execution')
        
        parser.add_argument('--encoder_initial_lr', type=float, default=1e-4, help='Initial learning rate for encoder (0: encoder frozen)')
        parser.add_argument('--bottleneck_initial_lr', type=float, default=1e-4, help='Initial learning rate for bottleneck (0: bottleneck frozen)')
        parser.add_argument('--decoder_initial_lr', type=float, default=1e-4, help='Initial learning rate for decoder (0: decoder frozen)')
        parser.add_argument('--encoder_weight_decay', type=float, default=0, help='Weight decay for encoder')
        parser.add_argument('--bottleneck_weight_decay', type=float, default=0, help='Weight decay for bottleneck')
        parser.add_argument('--decoder_weight_decay', type=float, default=0, help='Weight decay for decoder')
        parser.add_argument('--mlp_dropout', type=float, default=0.1, help='Dropout value for the Transformer MLP')
        parser.add_argument('--attn_dropout', type=float, default=0.1, help='Dropout value for the Transformer Attention')
        
        parser.add_argument('--dataset_folder', type=str, default='./../dataset', help='location of the downloaded and unzipped dataset')
        parser.add_argument('--preprocess_folder', type=str, default='./../preprocess', help='location where preprocesses images are stored')
        parser.add_argument('--out_folder', type=str, default='./../checkpoints', help='location where logs and checkpoints are stored')
        parser.add_argument('--save_checkpoint_frequency', type=int, default=1, help='After how many epochs the model and optimizer checkpoints should be saved')

        parser.add_argument('--use_s3_storage', action='store_true', default=False, help='whether to use s3 storage to store logs and checkpoints')
        parser.add_argument('--s3_prefix', type=str, default='train_out/test2/', help='prefix for storing s3 objects')
        return parser