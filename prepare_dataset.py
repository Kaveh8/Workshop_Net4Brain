"""
Build the analysis-ready workshop tables from raw LIBD Visium DLPFC data.

Outputs (into ./workshop_data/):
    dlpfc_gene_expression.csv   spots x  ~1200 HVGs   (log1p-normalised)
    dlpfc_metadata.csv          spots x  layer/donor/sample/coords
    scfea_input_counts.csv      genes x spots         (input for scFEA)
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.abspath("./.numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

SAMPLES = [151507, 151508, 151509, 151510, 151669, 151670,
           151671, 151672, 151673, 151674, 151675, 151676]
DONOR = {151507: "Br5292", 151508: "Br5292", 151509: "Br5292", 151510: "Br5292",
         151669: "Br5595", 151670: "Br5595", 151671: "Br5595", 151672: "Br5595",
         151673: "Br8100", 151674: "Br8100", 151675: "Br8100", 151676: "Br8100"}
N_HVG = 1000
N_PER_SAMPLE = 1000      # stratified subsample: keeps all 12 sections, all 3 donors
SEED = 0
OUT = "workshop_data"
os.makedirs(OUT, exist_ok=True)

lay = pd.read_csv("data/barcode_level_layer_map.tsv", sep="\t", header=None,
                  names=["barcode", "sample", "layer"])
lmap = lay.assign(key=lay["sample"].astype(str) + "_" + lay["barcode"]).set_index("key")["layer"]

ads = []
for s in SAMPLES:
    a = sc.read_10x_h5(f"data/{s}_filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    pos = pd.read_csv(f"data/{s}_tissue_positions_list.txt", header=None,
                      names=["barcode", "in_tissue", "row", "col", "py", "px"]).set_index("barcode")
    a.obs["sample_id"] = str(s)
    a.obs["donor"] = DONOR[s]
    a.obs["array_row"] = pos.loc[a.obs_names, "row"].values
    a.obs["array_col"] = pos.loc[a.obs_names, "col"].values
    a.obs["pxl_row"] = pos.loc[a.obs_names, "py"].values
    a.obs["pxl_col"] = pos.loc[a.obs_names, "px"].values
    a.obs["layer"] = [lmap.get(f"{s}_{b}", np.nan) for b in a.obs_names]
    a.obs_names = [f"{s}_{b}" for b in a.obs_names]
    ads.append(a)

adata = ad.concat(ads, join="inner")
adata = adata[~adata.obs["layer"].isna()].copy()
print("labelled spots x genes:", adata.shape, flush=True)

# stratified subsample, balanced across sections and layers, so the tables stay
# small enough to download and SHAP stays tractable in a workshop setting
rng = np.random.default_rng(SEED)
keep = []
for s, idx in adata.obs.groupby("sample_id", observed=True).indices.items():
    sub = adata.obs.iloc[idx]
    frac = N_PER_SAMPLE / len(sub)
    for lay_name, lidx in sub.groupby("layer", observed=True).indices.items():
        n = max(30, int(round(len(lidx) * frac)))
        n = min(n, len(lidx))
        keep.extend(sub.index[rng.choice(lidx, n, replace=False)])
adata = adata[sorted(set(keep))].copy()
print("after subsample:", adata.shape, flush=True)

# basic QC
sc.pp.filter_genes(adata, min_cells=50)
adata = adata[:, ~adata.var_names.str.startswith(("MT-", "RPL", "RPS"))].copy()
print("after QC:", adata.shape, flush=True)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key="donor")
hvg = adata.var_names[adata.var["highly_variable"]]
print("HVGs:", len(hvg), flush=True)

expr = pd.DataFrame(np.asarray(adata[:, hvg].X.todense()),
                    index=adata.obs_names, columns=hvg)
expr.round(4).to_csv(f"{OUT}/dlpfc_gene_expression.csv")

meta = adata.obs[["sample_id", "donor", "layer", "array_row", "array_col",
                  "pxl_row", "pxl_col"]].copy()
meta.to_csv(f"{OUT}/dlpfc_metadata.csv")

# scFEA input: genes x cells, raw counts, restricted to metabolic genes
mg = pd.read_csv("data/module_gene_m168.csv")
mod_genes = sorted({g for g in mg.iloc[:, 1:].values.ravel()
                    if isinstance(g, str) and g.strip()})
present = [g for g in mod_genes if g in adata.var_names]
print("scFEA metabolic genes present:", len(present), "/", len(mod_genes), flush=True)

cnt = pd.DataFrame(np.asarray(adata[:, present].layers["counts"].todense()).T,
                   index=present, columns=adata.obs_names)
cnt.to_csv(f"{OUT}/scfea_input_counts.csv")
print("wrote", OUT, flush=True)
for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 1), "MB")
