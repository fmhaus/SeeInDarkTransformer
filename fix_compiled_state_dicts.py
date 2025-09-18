import torch

FILE = './../training/co_adapt_1/comp/model_checkpoint_30.pt'

if __name__ == '__main__':
    state_dict = torch.load(FILE, map_location=torch.device('cpu'), weights_only=True)

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    torch.save(new_state_dict, FILE + '_fixed.pt')
    print(f'saved to {FILE + '_fixed.pt'}')