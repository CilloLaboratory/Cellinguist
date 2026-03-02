from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd
import torch


def export_perceiver_gene_embeddings(
    checkpoint_path: str,
    out_tsv_gz: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    train_cfg = ckpt.get("config", {})
    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    if encoder_type != "perceiver":
        raise ValueError(
            f"Checkpoint encoder_type is '{encoder_type}', expected 'perceiver'."
        )

    genes_common = ckpt.get("genes_common", None)
    if genes_common is None:
        raise ValueError("Checkpoint does not contain 'genes_common'.")

    state_dict = ckpt.get("model_state_dict", {})

    weight = None
    for key in ("encoder.gene_embedding.weight", "module.encoder.gene_embedding.weight"):
        if key in state_dict:
            weight = state_dict[key]
            break

    if weight is None:
        raise ValueError(
            "Could not find perceiver gene embedding weights in checkpoint state_dict."
        )

    gene_emb = weight.detach().cpu().numpy()
    if gene_emb.shape[0] != len(genes_common):
        raise ValueError(
            "Mismatch between checkpoint genes_common and gene embedding rows: "
            f"{len(genes_common)} vs {gene_emb.shape[0]}"
        )

    df = pd.DataFrame(
        gene_emb,
        columns=[f"dim_{i+1}" for i in range(gene_emb.shape[1])],
    )
    df.insert(0, "gene", list(genes_common))

    out_path = Path(out_tsv_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        df.to_csv(f, sep="\t", index=False)

    print(f"[export] Wrote Perceiver gene embeddings to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Perceiver gene token embeddings from a VAE checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint (.ckpt).")
    parser.add_argument("--out", required=True, help="Output .tsv.gz path for gene embeddings.")
    args = parser.parse_args()

    export_perceiver_gene_embeddings(
        checkpoint_path=args.checkpoint,
        out_tsv_gz=args.out,
    )


if __name__ == "__main__":
    main()
