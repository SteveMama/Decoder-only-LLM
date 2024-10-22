from model import *
from main import *
from train import *


name = 'Llama3_2024-04-19 | 15-18-16'

with open(f' {name}.json', 'r') as f:
    params_dict = json.load(f)

params = ModelArgs(**params_dict)
params.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Llama3(params, tokenizer).to(params.device)

path = f'{name}.pth'

model.load_state_dict(torch.load(path))

print(sum(p.numel() for p in model.parameters())/1e3, 'K paramaters')

model.eval()