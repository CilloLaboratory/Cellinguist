from torch.utils.data import DataLoader
from cellinguist_cbow.data.datasets import SingleCellDataset
from cellinguist_cbow.models.cbow import (
    CBOWModel,
    CBOWTrainer,
    CBOWTrainerConfig,
    build_cbow_training_pairs,
)

# 1. Build SingleCellDataset
sc_ds = SingleCellDataset("path/to.h5ad")
token_id_seqs = sc_ds._token_id_seqs  # or expose via a property

# 2. Build CBOW triples
target_ids, context_ids, negative_ids = build_cbow_training_pairs(
    token_id_seqs=token_id_seqs,
    vocab_size=sc_ds.vocab_size,
    window_size=5,
    num_negatives=10,
)

# 3. Wrap in a TensorDataset/DataLoader for training
cbow_dataset = torch.utils.data.TensorDataset(target_ids, context_ids, negative_ids)
cbow_loader = DataLoader(cbow_dataset, batch_size=1024, shuffle=True)

# 4. Model + trainer
model = CBOWModel(vocab_size=sc_ds.vocab_size, emb_dim=128)
config = CBOWTrainerConfig(lr=1e-3, epochs=5, device="cuda")
trainer = CBOWTrainer(model, config)

for epoch in range(config.epochs):
    loss = trainer.train_epoch(
        {"target_ids": b[0], "context_ids": b[1], "negative_ids": b[2]}
        for b in cbow_loader
    )
    print(f"Epoch {epoch}: loss={loss:.4f}")

trainer.save_embeddings("gene_token_embeddings.pt")
