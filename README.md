# Multimodal Machine Learning on Spatial Transcriptomics

**Net4Brain Training School — hands-on workshop material**

A three-day, end-to-end machine-learning pipeline on spatial transcriptomics data, written for
students **without a computer-science background**. Plots use **scanpy** one-liners; the model
itself is scikit-learn. Everything runs in Google Colab; no local installation and no GPU are
needed.

The workshop combines two modalities for every tissue spot:

| Modality | What it is | Provenance |
|---|---|---|
| **Gene expression** | Highly variable genes per spot | 10x Genomics Visium — **measured** |
| **Metabolic flux** | 168 metabolic module rates per spot | scFEA — **inferred from the expression data** |

That second row is a caveat carried throughout the material: the fluxes are *estimated from the
transcriptome*, not measured with a metabolic assay. The notebooks state this every time they
discuss a metabolic result.

## The two datasets

Steps are **demonstrated** on healthy cortex, then **repeated by students** on tumour tissue.
Section numbers match across the three notebooks, so students copy a demo cell and change names
(`layer` → `region`, `donor` → `patient`, file prefixes, marker genes).

| | Demonstration | Student mini-project |
|---|---|---|
| **Tissue** | Healthy human DLPFC | Human glioblastoma, IDH-wildtype |
| **Spots** | 12,000 | 35,187 |
| **Subjects** | 3 donors, 12 sections | 6 patients, 15 sections |
| **Genes** | 1,000 highly variable | 2,000 highly variable |
| **Flux modules** | 168 | 168 |
| **Prediction task** | Superficial (L1–L3) vs deep (L4–L6) layers | Tumour core vs periphery |
| **Grouping for evaluation** | Donor | Patient |
| **Notebook** | `Demo_DLPFC.ipynb` | `Try_GBM.ipynb` |

## Notebooks

| Notebook | Who runs it | What it is |
|---|---|---|
| **`Demo_DLPFC.ipynb`** | Instructor | The complete pipeline, demonstrated end to end on healthy cortex. |
| **`Try_GBM.ipynb`** | Students | The same numbered sections on glioblastoma, as **TASK cells** with hints and expected variable names. |
| **`Solution_GBM.ipynb`** | Both | Every task completed. For checking student work, or to show as the answer key. |

Each notebook is divided into **Day 1 / Day 2 / Day 3** sections matching the schedule below.
Days are sequential: Day 2 continues from the Day 1 variables, and Day 3 continues from Day 2.
If a Colab session disconnects, re-run the earlier cells.

Day 1 is plots. Day 2 is a short train/test split, a logistic classifier, and a
three-way comparison of genes vs flux vs both. Day 3 explains that model with a
SHAP beeswarm plot and the logistic coefficients.

## Running the notebooks

### On Google Colab (recommended for students)

1. Open <https://colab.research.google.com> → **GitHub** → paste this repository URL.
2. Open the notebook you want.
3. Uncomment and run `!pip install -q scanpy shap` in the first code cell, then run the setup
   cell. The next cell downloads the data tables from this repository (about 48 MB for the
   glioblastoma data).

### Locally

```bash
git clone https://github.com/Kaveh8/Workshop_Net4Brain.git
cd Workshop_Net4Brain
pip install numpy pandas matplotlib scanpy scikit-learn shap jupyter
jupyter notebook
```

The data is committed to this repository, so cloning gives you everything. The download cell
detects the files are already present and skips the download.

## The three days

| Day | Title | Content |
|---|---|---|
| **1** | Introduction & Preparing Data | Load the tables, pack them into AnnData, and explore with coloured bar, spatial, and violin plots. |
| **2** | Training models | Hold out one subject, train a logistic classifier, read the classification report and confusion matrix, then compare genes vs flux vs both. |
| **3** | Explainable AI | SHAP beeswarm plot and logistic coefficients — no data reload. |

Students present their results on the final day.

## Data

The `dataset/` folder contains everything the notebooks need. All tables are gzipped CSVs with
the spot ID in the first column, and **the three tables for each dataset share the same row
index** — that alignment is what makes the analysis multimodal.

| File | Contents |
|---|---|
| `dlpfc_gene_expression.csv.gz` | 12,000 spots × 1,000 genes, log-normalised |
| `dlpfc_flux.csv.gz` | 12,000 spots × 168 metabolic modules |
| `dlpfc_metadata.csv.gz` | Section, donor, layer annotation, spot coordinates |
| `gbm_gene_expression.csv.gz` | 35,187 spots × 2,000 genes, log-normalised |
| `gbm_flux.csv.gz` | 35,187 spots × 168 metabolic modules |
| `gbm_metadata.csv.gz` | Patient, section, region, detailed region, spot coordinates |
| `scfea_module_info.csv` | Module ID → reaction, so `M_40` can be named |

### How the metabolic flux was generated

**The flux values were not measured.** Both datasets are Visium spatial transcriptomics, so the
only molecule assayed at each spot is mRNA. The flux tables were computed from the gene
expression with [scFEA](https://github.com/changwn/scFEA), which we ran before the workshop; the
notebooks name this modality *inferred flux* throughout and the students are asked to do the
same.

scFEA represents human central metabolism as a graph of metabolites and reactions, collapses it
into 168 modules (each a short chain of consecutive reactions), maps each module to the genes
encoding its enzymes, and trains a small neural network to predict a per-module flux from those
genes. Its loss penalises imbalance of intermediate metabolites — flow in must match flow out —
which is what makes the output more than a rescaled average of the module's genes.

The pipeline we ran, starting from the raw count matrices of the two published datasets:

```bash
git clone https://github.com/changwn/scFEA.git

# input: a genes x spots raw count matrix as CSV
python src/scFEA.py \
  --input_dir  <dir containing the count matrix> \
  --test_file  <counts.csv> \
  --moduleGene_file       module_gene_m168.csv \
  --stoichiometry_matrix  cmMat_c70_m168.csv \
  --cName_file            cName_c70_m168.csv \
  --sc_imputation True \
  --train_epoch 30 \
  --output_flux_file      <flux.csv>
```

`module_gene_m168.csv` maps genes to modules; `cmMat_c70_m168.csv` is the stoichiometry matrix
(70 compounds × 168 modules). The human M168 module set was used for both datasets. The
glioblastoma data was processed in per-patient batches for memory reasons and the outputs
concatenated. In both cases the resulting flux table was reindexed onto the same spot order as
the expression and metadata tables, and that alignment is asserted in the notebooks.

`dataset/scfea_module_info.csv` is scFEA's own module annotation file
(`Human_M168_information.symbols.csv`), shipped locally so the Day 3 module-to-reaction lookup
does not need network access.

**A consequence worth stating in the students' presentations:** because the flux is derived from
the expression, the two modalities are not independent measurements. A gene and a flux module
appearing together near the top of a feature ranking is one piece of evidence, not two. This is
also the most likely reason fusing the two modalities does not beat gene expression alone in
either dataset.

### A note on the glioblastoma labels

The `region` column has two values, `Core` and `Periphery`. **`Periphery` does not mean healthy
tissue** — glioblastoma infiltrates well beyond its visible core, so peripheral spots still
contain tumour cells. The contrast is *core versus surrounding tissue*, not *tumour versus
normal*.

`Periphery` is also a merged class. The finer original annotation is kept in the
`region_detailed` column, so nothing is lost by the merge and students can return to it.

## Sources and citation

**DLPFC spatial transcriptomics** — Maynard KR, Collado-Torres L, Weber LM, et al. (2021)
*Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex.*
Nature Neuroscience 24:425–436. <https://doi.org/10.1038/s41593-020-00787-0>
Data: <https://github.com/LieberInstitute/HumanPilot>

**Glioblastoma spatial transcriptomics** — Ravi VM, Will P, Kueckelhaus J, et al. (2022)
*Spatially resolved multi-omics deciphers bidirectional tumor-host interdependence in
glioblastoma.* Cancer Cell 40:639–655. <https://doi.org/10.1016/j.ccell.2022.05.009>
Region annotations follow the Ivy Glioblastoma Atlas Project scheme
(<https://glioblastoma.alleninstitute.org>).

**Metabolic flux inference** — Alghamdi N, Chang W, Dang P, et al. (2021) *A graph neural network
model to estimate cell-wise metabolic flux using single-cell RNA-seq data.* Genome Research
31:1867–1884. <https://doi.org/10.1101/gr.271205.120>
Software: <https://github.com/changwn/scFEA>

**SHAP** — Lundberg SM, Lee S-I (2017) *A unified approach to interpreting model predictions.*
NeurIPS 30. <https://github.com/shap/shap>

The tables here are processed subsets prepared for teaching: spots subsampled, genes reduced to
the most variable, and values rounded to keep the download small. For research use, go to the
original sources above.
