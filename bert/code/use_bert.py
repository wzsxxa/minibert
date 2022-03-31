import torch
import data_process

# net1 = torch.load('bert50.pth')
# net2 = torch.load('bert100.pth')
# net3 = torch.load('bert150.pth')

tokens_a, token_b = ['what', 'a', 'nice', 'day'] , ['i', 'am', 'very', '<mask>']
tokens, segments  = data_process.get_tokens_and_segments(tokens_a, token_b)
# print(tokens[9])
tokens_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
