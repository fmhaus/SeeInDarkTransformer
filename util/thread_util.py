import torch
from tqdm import tqdm

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, userdata, process_fn):
        self.tasks = tasks
        self.userdata = userdata
        self.process_fn = process_fn
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return_val = self.process_fn(self.userdata, self.tasks[idx])
        if return_val is not None:
            return idx, return_val
        else:
            return ()

def process_parallel(tasks, userdata, process_fn, num_workers, tqdm_title):
    dataset = TaskDataset(tasks, userdata, process_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False
        )
    
    results = [None] * len(dataset)
    
    if tqdm_title:
        iter = tqdm(dataloader, tqdm_title)
    else:
        iter = dataloader
    
    for item in iter:
        if len(item) == 2:
            i, res = item
            results[i] = res
    
    return results