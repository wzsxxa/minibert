import d2l
import data_process
import torch
from data_process import get_tokens_and_segments

net3 = torch.load('bert150.pth')
train_iter, vocab = data_process.load_data_wiki(batch_size=512, max_len=64)
tokens_a, token_b = ['what', 'a', 'nice', 'day'], ['i', 'am', 'very', '<mask>']
tokens, segments = data_process.get_tokens_and_segments(tokens_a, token_b)
for i in range(53):
    token_b += ['<pad>']
print(tokens)
print(segments)
tokens_ids = torch.tensor(vocab[tokens], device='cuda').unsqueeze(0)
segments = torch.tensor(segments, device='cuda').unsqueeze(0)
valid_len = torch.tensor(11, device='cuda').unsqueeze(0)
pred_positions = torch.tensor([9],device='cuda').unsqueeze(0)
# _, _, d1 = net1(tokens_ids, segments, valid_len, pred_positions)
# _, _, d2 = net2(tokens_ids, segments, valid_len, pred_positions)
_, _, d3 = net3(tokens_ids, segments, valid_len, pred_positions)

# idx1 = torch.argmax(d1[0][0], dim = 0).item()
# print('bert50 predicition: what a nice day, i am very '+ vocab.to_token(idx1))
#
# idx2 = torch.argmax(d2[0][0], dim=0).item()
# print('bert100 predicition: what a nice day, i am very ' + vocab.to_token(idx2))

idx3 = torch.argmax(d3[0][0], dim=0).item()
print('bert150 predicition: what a nice day, i am very ' + vocab.to_token(idx3))
