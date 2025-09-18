import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

LOGS_FOLDER = './../training/co_adapt_1/logs'
TITLE = 'Learning (Co-Adapt, 1)'

if __name__ == '__main__':
    
    logs = []
    for filename in os.listdir(LOGS_FOLDER):
        # test for log_{integer}.json
        match = re.search(r'^log_(\d+)\.json$', filename)
        if match:
            epoch = int(match.group(1))
            with open(os.path.join(LOGS_FOLDER, filename), 'r') as fr:
                log = json.load(fr)
            index = epoch-1
            if len(logs) <= index:
                logs = logs + [None] * (index + 1 - len(logs))
            
            logs[index] = log
    
    x = np.arange(1, len(logs) + 1)
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(TITLE)
    
    print(logs[0])
    
    def show_graphs(index, title, keys):
        ax = axes[*index]
        for key in keys:
            ax.plot(x, [log[key] for log in logs], label=key)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    
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
    