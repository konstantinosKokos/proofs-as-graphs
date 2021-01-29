from .pretrain import *
from ...data.process_proofnets import load_processed
from ..utils.batching import make_dataloader, graph_to_data
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


(train, dev, test), atom_map, embedding_table = load_processed()
train_dl = make_dataloader([graph_to_data(*graph) for graph in train], 128)
dev_dl = make_dataloader([graph_to_data(*graph) for graph in dev], 128)
test_dl = make_dataloader([graph_to_data(*graph) for graph in test], 128)

model = Model(atom_map, embedding_table)
logger = PretrainLogger(CrossEntropyLoss(ignore_index=0, reduction='sum'), Adam(model.parameters(), lr=1e-03))


for i in range(100):
    print(i)
    log_epoch(model=model, dataloader=train_dl, mask_chance=0.5, num_hops=12, logger=logger, train=True)
    print(logger.aggr_last_epoch(True))
    log_epoch(model=model, dataloader=dev_dl, mask_chance=0.5, num_hops=12, logger=logger, train=False)
    print(logger.aggr_last_epoch(False))
