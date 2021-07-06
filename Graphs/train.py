import torch

from .preprocessing.process_sick import proc_sick
from .neural.tokenizer import BertWrapper
from .neural.utils.batching import graphs_to_data, pair_loader
from .neural.utils.metrics import per_class_accuracy, epoch_accuracies, mean
from .neural.model import PairClassifier

from torch_optimizer.adabound import AdaBound
from torch.nn import CrossEntropyLoss


bw = BertWrapper()
sents, samples, node_vocab, edge_vocab = proc_sick('/home/kokos/Projects/graphs/Graphs/npn_nets_2.p', bw.words_to_ids)


train_data, dev_data, test_data = [], [], []
for _, p, h, label, subset in samples:
    sample = graphs_to_data(sents[p][1], sents[h][1], label)
    append_to = train_data if subset == 'TRAIN' else test_data if subset == 'TEST' else dev_data
    append_to.append(sample)
# assert len(train_data) + len(dev_data) + len(test_data) == len(samples)

train_dl = pair_loader(train_data, 64, True)
dev_dl = pair_loader(dev_data, 128, False)
test_dl = pair_loader(test_data, 128, False)

network = PairClassifier([32, 64, 128, 256, 128, 64], 32, len(node_vocab), len(edge_vocab),
                         [(3, 1), (4, 0)], bw, num_heads=4).cuda()
for p in network.bert.model.parameters():
    p.requires_grad = False
num_epochs = 500

opt = AdaBound(network.parameters(), lr=5e-5, betas=(0.9, 0.98), final_lr=1e-03,
               gamma=1e-4, eps=1e-8, weight_decay=1e-2, amsbound=False)
loss_fn = CrossEntropyLoss(weight=torch.tensor([0.33, 1, 1], device='cuda'))

for epoch in range(num_epochs):
    print('=' * 64)
    print(epoch)
    e_loss = []
    e_accuracies = []
    network.train()
    for batch in train_dl:
        batch = batch.cuda()
        preds = network.batch_to_label(batch)
        loss = loss_fn(preds, batch.y)
        e_accuracies.append(per_class_accuracy(preds, batch.y))
        loss.backward()
        opt.step()
        opt.zero_grad()
        e_loss.append(loss.item())
    print(f'\t{mean(e_loss)}')
    print('\t' + '\t'.join(["{:.2f}".format(acc * 100) for acc in epoch_accuracies(e_accuracies)]))
    print('\t' + '-' * 60)
    with torch.no_grad():
        network.eval()
        e_loss = []
        e_accuracies = []
        for batch in test_dl:
            batch = batch.cuda()
            preds = network.batch_to_label(batch)
            loss = loss_fn(preds, batch.y)
            e_loss.append(loss.item())
            e_accuracies.append(per_class_accuracy(network.batch_to_label(batch), batch.y))
        print(f'\t{mean(e_loss)}')
        print('\t' + '\t'.join(["{:.2f}".format(acc * 100) for acc in epoch_accuracies(e_accuracies)]))