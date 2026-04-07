# `construct-grn` Algorithm Specification (Implemented v1)

## 1. Scope

`construct-grn` builds TF->target directed GRNs from a trained Cellinguist VAE checkpoint.

Current implementation supports:
- Global GRN (`out_dir/global`)
- Optional context-specific GRNs by `adata.obs[context_key]` (`out_dir/contexts/<context>`)
- Edge confidence from optional bootstrap resampling

## 2. Interface

Command:

```bash
construct-grn --config /path/to/grn_construct.yml
```

The CLI currently accepts config-file mode only (`--config`).

## 3. Required and Optional Config

Required:
- `adata_path`
- `checkpoint_path`
- `tf_list_path` (newline-delimited TF symbols)
- `out_dir`

Common optional:
- `gene_key` (default `gene`)
- `layer`
- `batch_key` / `cond_key`
- `context_key`
- `min_cells_per_context` (default `300`)
- `max_cells`, `max_cells_seed`
- `device` (default `cuda`)
- `batch_size`, `num_workers`, `backed`
- `token_index_cache_dir`, `token_index_cache_require`, `transformer_precompute_token_indices`
- `prior_edges_tsv` (`tf,target,prior_weight`)

Scoring/selection:
- `score_min` (default `0.35`)
- `top_k_per_tf` (default `50`)
- `min_abs_effect` (default `0.01`)
- `allow_self_edges` (default `false`)

Perturbation/sign:
- `perturb_frac` (default `0.10`)
- `perturb_min_abs` (default `0.25`)
- `sign_eps` (default `1e-3`)
- `sign_consistency_min` (default `0.60`)

Fusion:
- `w_a` (default `0.25`)
- `w_e` (default `0.15`)
- `w_d` (default `0.55`)
- `w_p` (default `0.05`)

Bootstrap:
- `bootstrap_iters` (default `0`)
- `bootstrap_cell_frac` (default `0.80`)

Reproducibility:
- `eps` (default `1e-8`)
- `seed` (default `0`)

## 4. Data and Model Alignment

1. Load checkpoint and `genes_common`.
2. Build `SingleCellVAEDataset` aligned to checkpoint gene order.
3. Rebuild encoder/decoder architecture from checkpoint `config` (`cbow`, `perceiver`, `transformer` supported).
4. Load weights non-strictly (ignores `batch_adversary.*` export incompatibility keys).
5. Resolve TF indices by exact gene-name match; unmatched TFs are recorded in metadata.

## 5. Scoring Components

Notation:
- `G`: number of genes
- `T`: number of matched TFs
- `x`: observed expression vector
- `z = mu_z(x)`: deterministic encoder latent (no sampling)
- `mu`: decoder mean output (`ZINBExpressionDecoder` returns `mu, theta, pi`; GRN uses `mu`)

### 5.1 Attention evidence `A` (Transformer only)

Implemented via `TransformerCellEncoder.forward_with_attention(...)`.

Per batch:
- Extract per-layer/head attention tensor and token->gene mapping.
- Mean-reduce attention across layers and heads.
- For each TF token present in a cell, accumulate attention from TF token to all valid gene tokens in that cell.

Aggregate:
- `A_raw(t,g) = sum_attn / count`
- Row-wise robust scaling per TF:
  - `A = clip((A_raw - q05)/(q95 - q05 + eps), 0, 1)`

If attention unavailable (non-transformer), `A=0` matrix.

### 5.2 Embedding evidence `E`

Uses encoder `gene_embedding.weight` when present.

For each TF-target pair:
- cosine similarity of normalized gene embeddings
- map to `[0,1]` via `(cos + 1) / 2`
- row-wise rank normalization over targets:
  - `E(t,g) = rank / (G-1)`

If no gene embedding exists, `E=0` matrix.

### 5.3 Decoder perturbation evidence `D` and signed effect

For each TF `t`:
1. Define eligible cells as `x[:, t] > 0`.
2. Build perturbed input:
   - `x'_t = x_t + max(perturb_min_abs, perturb_frac * max(x_t, 1.0))`
3. Re-encode and decode: `z'`, `mu'`.
4. Compute per-cell delta:
   - `d = log1p(mu') - log1p(mu_base)`

Aggregate per `(t,g)`:
- `effect_size(t,g)` = mean of `d` over eligible cells
- `p_pos(t,g)` = fraction of cells with `d > sign_eps`
- `p_neg(t,g)` = fraction of cells with `d < -sign_eps`

Sign label:
- `activation` if `effect_size > sign_eps` and `p_pos >= sign_consistency_min`
- `repression` if `effect_size < -sign_eps` and `p_neg >= sign_consistency_min`
- else `ambiguous`

Decoder magnitude score:
- `D_raw = abs(effect_size)`
- row-wise robust scaling per TF to `[0,1]` (same q05/q95 scheme)

### 5.4 Prior evidence `P`

If `prior_edges_tsv` provided:
- load `tf,target,prior_weight`
- clip `prior_weight` to `[0,1]`
- unmatched TF/target rows are ignored

Else `P=0`.

## 6. Fusion and Selection

Base fusion:
- `S = w_a*A + w_e*E + w_d*D + w_p*P`

If attention unavailable:
- set `w_a=0`, renormalize `w_e,w_d,w_p` to sum to 1.

Edge filtering:
1. `S >= score_min`
2. `abs(effect_size) >= min_abs_effect`
3. remove self-edges unless `allow_self_edges=true`
4. keep top `top_k_per_tf` edges per TF (if `top_k_per_tf > 0`)

## 7. Bootstrap Confidence

If `bootstrap_iters > 0`:
1. Sample selected cell indices with replacement (`bootstrap_cell_frac`).
2. Recompute `A/D` (reuse `E`), refusion, reselection.
3. Count selection frequency per edge.

`bootstrap_freq = selected_count / bootstrap_iters`.

If `bootstrap_iters == 0`:
- `bootstrap_freq` is binary (`1.0` if selected in full run, else `0.0`).

Confidence labels in edge table:
- `high` if `bootstrap_freq >= 0.80`
- `medium` if `0.50 <= bootstrap_freq < 0.80`
- `low` otherwise

## 8. Outputs

Top-level:
- `out_dir/run_metadata.json`
- `out_dir/qc_metrics.json` (list of per-context QC dicts)

Per network directory (`global` and each eligible context):
- `edges.tsv.gz`
- `nodes.tsv.gz`
- `edge_components.npz`
- `qc_metrics.json`

### 8.1 `edges.tsv.gz` columns

- `tf`
- `target`
- `score`
- `effect_size`
- `sign` (`activation|repression|ambiguous`)
- `attention_score`
- `embedding_score`
- `decoder_score`
- `prior_score`
- `bootstrap_freq`
- `confidence`
- `n_cells_used`
- `context`

### 8.2 `nodes.tsv.gz` columns

- `gene`
- `is_tf` (`0/1`)

### 8.3 `edge_components.npz` arrays

- `A`, `E`, `D`, `P`, `S`
- `effect_size`
- `sign_code` (`-1,0,1`)
- `tf_names`
- `gene_names`

## 9. QC/Metadata Fields (current)

Per-network `qc_metrics.json` includes:
- `context`
- `n_cells_total`, `n_cells_used`
- `n_genes`, `n_tfs_matched`
- `edge_count_final`
- `fraction_activation`, `fraction_repression`, `fraction_ambiguous`
- `median_bootstrap_freq`
- `attention_available`
- `baseline_forward_s`, `perturb_s`, `bootstrap_s`, `total_s`

Top-level `run_metadata.json` includes:
- full resolved config
- checkpoint train config
- `encoder_type`
- `genes_common_n`
- `n_tfs_requested`, `n_tfs_matched`, `missing_tfs`
- `git_commit` (if available)
- `run_total_s`

## 10. Transformer Attention Export Status

Implemented in v1:
- `TransformerCellEncoder.forward_with_attention(...)`
- returns `mu`, `logvar`, and extras with:
  - `attn_weights`
  - `token_gene_idx`
  - `token_gene_mask`
  - `key_padding_mask`

This replaces the earlier “planned but not implemented” status.
