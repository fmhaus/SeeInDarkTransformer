import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

LOGS_FOLDER = './../training/train_6/logs'
TITLE = 'SID Bottleneck transformer (2 Blocks C)'

LOSS_BENCHMARK = 0.030083592981100082
PSNR_BENCHMARK = 28.586631774902344
BENCHMARK_LABEL = 'sid_original_val'

class LogsData:
    def __init__(self, path, name = None):
        self.name = name
        logs = []
        for filename in os.listdir(path):
            # test for log_{integer}.json
            match = re.search(r'^log_(\d+)\.json$', filename)
            if match:
                epoch = int(match.group(1))
                with open(os.path.join(path, filename), 'r') as fr:
                    log = json.loads(fr.read())
                index = epoch-1
                if len(logs) <= index:
                    logs = logs + [None] * (index + 1 - len(logs))
                
                logs[index] = log
        
        self.x = np.array([i for i, log in enumerate(logs) if log is not None])
        self.logs = [log for log in logs if log is not None]
    
  
    def add_graph(self, axes, plot_index, title, list_fns, labels):
        ax = axes[*plot_index]
        for list_fn, label in zip(list_fns, labels):
            if self.name:
                label = f'{self.name}_{label}'
            ax.plot(self.x, [list_fn(log) for log in self.logs if log], label=label)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        
        """
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
        """

class Visuals:
    def __init__(self, title):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2)
        self.fig.suptitle(title)
        self.x = []
    
    def add_logs(self, logs_data):
        logs_data.add_graph(self.axes, (0, 0), 'Loss', 
            [lambda log: log['avg_train_loss'], lambda log: log['avg_val_loss']],
            ['avg_train_loss', 'avg_val_loss'])
    
        logs_data.add_graph(self.axes, (0, 1), 'Time', 
            [lambda log: log['train_time'], lambda log: log['val_time']],
            ['train_time', 'val_time'])
        
        logs_data.add_graph(self.axes, (1, 0), 'PSNR', 
            [lambda log: log['avg_val_psnr']],
            ['avg_val_psnr'])
        
        logs_data.add_graph(self.axes, (1, 1), 'Learning rates', 
            [lambda log: log['learning_rates'][0], lambda log: log['learning_rates'][1], lambda log: log['learning_rates'][2]],
            ['encoder', 'bottleneck', 'decoder'])
        
        self.x = list(set(self.x) | set(logs_data.x))
        self.x.sort()

    def add_benchmarks(self, benchmark_loss, benchmark_psnr, label):
        self.axes[0, 0].plot(self.x, np.full(len(self.x), benchmark_loss), label=label, color='gray', linestyle='--')
        self.axes[1, 0].plot(self.x, np.full(len(self.x), benchmark_psnr), label=label, color='gray', linestyle='--')
    
    def show(self):
        self.axes[0, 0].legend()
        self.axes[0, 1].legend()
        self.axes[1, 0].legend()
        self.axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    
    train_1 = LogsData('./../training/co_adapt_1/logs', '4b_c')
    train_3 = LogsData('./../training/co_adapt_3/logs', '3B')
    train_4 = LogsData('./../training/finetune_4/logs', '3B_f')
    train_5 = LogsData('./../training/train_5/logs', '2B')
    train_6 = LogsData('./../training/train_6/logs', '2B_c')
    
    vis = Visuals('2B vs 2B_c')
    vis.add_logs(train_1)
    vis.add_logs(train_3)
    vis.add_logs(train_4)
    vis.add_logs(train_5)
    vis.add_logs(train_6)
    vis.add_benchmarks(LOSS_BENCHMARK, PSNR_BENCHMARK, BENCHMARK_LABEL)
    
    vis.show()
    