import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

LOGS_FOLDER = './../training/co_adapt_3/logs'
TITLE = 'SID Bottleneck transformer (3 Blocks, Co-adapt, v3)'

LOSS_BENCHMARK = 0.030083592981100082
PSNR_BENCHMARK = 28.586631774902344
BENCHMARK_LABEL = 'sid_original_val'

if __name__ == '__main__':
    
    logs = []
    for filename in os.listdir(LOGS_FOLDER):
        # test for log_{integer}.json
        match = re.search(r'^log_(\d+)\.json$', filename)
        if match:
            epoch = int(match.group(1))
            with open(os.path.join(LOGS_FOLDER, filename), 'r') as fr:
                log = json.loads(fr.read())
            index = epoch-1
            if len(logs) <= index:
                logs = logs + [None] * (index + 1 - len(logs))
            
            logs[index] = log
    
    x = np.array([i for i, log in enumerate(logs) if log is not None])
    logs = [log for log in logs if log is not None]
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(TITLE)
    
    def show_graphs(index, title, keys):
        ax = axes[*index]
        for key in keys:
            ax.plot(x, [log[key] for log in logs], label=key)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        
    if LOSS_BENCHMARK is not None:
        axes[0, 0].plot(x, np.full((len(logs)), LOSS_BENCHMARK), label=BENCHMARK_LABEL, color='gray', linestyle='--')

    if PSNR_BENCHMARK is not None:
        axes[1, 0].plot(x, np.full((len(logs)), PSNR_BENCHMARK), label=BENCHMARK_LABEL, color='gray', linestyle='--')
    
    show_graphs((0, 0), 'Loss', ['avg_train_loss', 'avg_val_loss'])
    show_graphs((0, 1), 'Time', ['train_time', 'val_time'])
    show_graphs((1, 0), 'PSNR', ['avg_val_psnr'])
    
    ax_lr = axes[1, 1]
    ax_lr.plot(x, [log['learning_rates'][0] for log in logs], label='Encoder')
    ax_lr.plot(x, [log['learning_rates'][1] for log in logs], label='Bottleneck')
    ax_lr.plot(x, [log['learning_rates'][2] for log in logs], label='Decoder')
    ax_lr.set_title('Learning rates')
    ax_lr.set_xlabel('Epoch')
    ax_lr.legend()
    
    plt.tight_layout()
    plt.show()
    