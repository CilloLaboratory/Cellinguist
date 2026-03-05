from cellinguist.data.datasets import SingleCellDataset
from torch.utils.data import DataLoader

ds = SingleCellDataset(
    "/home/arc85/Desktop/Cellinguist/cytokine_dictionary_2k_pbs_ifng_251108.h5ad",
    gene_key="gene",
    cond_key=None,     # or None
    n_bins=20,
    shuffle_tokens=True,
    min_expr=0.0,
)

dl = DataLoader(ds, batch_size=1, shuffle=True)

data_iterator = iter(dl)
first_batch = next(data_iterator)

# Check out items in the returned dict
for key, value in first_batch.items():
    print(f"Key: '{key}', Class: {type(value)}")

print(" token_ids shape:", first_batch["token_ids"].shape)
print(" tokens length:", len(first_batch["tokens"]))
print(" lib size", first_batch["libsize"])
print("  cond_idx:", first_batch["cond_idx"])
print("  cell_idx:", first_batch["cell_idx"])