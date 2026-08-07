# Mini-Project: Multimodal Machine Learning on the Human Brain Cortex

**Training School — hands-on mini-project (Days 1–2, presented on the final day)**

## Aim

Each group builds a small end-to-end machine-learning pipeline that combines **two modalities**
measured on the same piece of tissue — *spatial gene expression* and *estimated metabolic flux* —
and asks a single, concrete biological question:

> **Which genes and which metabolic reactions distinguish the layers of the human cerebral cortex?**

The cortex is organised into six neuronal layers plus underlying white matter. These layers differ
in cell type, connectivity and energy demand, so they give us a clean, well-understood ground truth
against which a model's predictions — and its explanations — can be judged. No prior coding
experience is assumed: every step is demonstrated during the training sessions, and participants
repeat and adapt it on the provided notebook.

## Dataset

**LIBD Human Dorsolateral Prefrontal Cortex (DLPFC), 10x Genomics Visium**
(Maynard *et al.*, *Nature Neuroscience*, 2021) — 12 tissue sections from 3 neurotypical adult
donors, ~47,700 spatial spots, each with a manually curated layer label (**L1–L6** and **white
matter**) and *x, y* tissue coordinates.

- Data & annotations: https://github.com/LieberInstitute/HumanPilot
- R/Bioconductor package and browser: https://bioconductor.org/packages/spatialLIBD

We will supply a **pre-processed, analysis-ready table** so that no time is lost to data
wrangling: one row per spot, columns = normalised expression of ~2,000 highly variable genes,
**≈170 metabolic-module flux values pre-computed with scFEA** (https://github.com/changwn/scFEA),
plus the layer label, donor ID and spatial coordinates. The flux-generation step is demonstrated
live in the metabolic-modelling session; participants receive the output rather than having to
run it themselves.

## What each group does

1. **Exploratory data analysis (your choice).** Describe the data before modelling it: class
   balance across layers, distribution and variance of genes vs. fluxes, correlation between the
   two modalities, a low-dimensional embedding (PCA/UMAP) coloured by layer and by donor, and at
   least one plot of a feature mapped back onto the tissue coordinates.
2. **Train and evaluate a model (your choice).** Predict the cortical layer of each spot. Fit at
   least one interpretable baseline (e.g. penalised logistic regression) and one tree-based model
   (e.g. random forest or gradient boosting). Crucially, run it **three times** — on genes alone,
   on fluxes alone, and on the two concatenated (*early integration*) — and report whether the
   combination actually helps. Hold out one **donor** entirely as the test set, so performance
   reflects generalisation to a new individual. Report balanced accuracy and a confusion matrix,
   and comment on which layers are confused with which.
3. **Explain the model with SHAP.** Compute SHAP values on the integrated model and identify the
   genes and the metabolic modules that drive each layer's prediction. Then take your top few
   features back to the tissue: plot them spatially and check whether the pattern the model relies
   on is a genuine biological gradient or an artefact.

## Deliverable

A **10-minute group presentation** on the final day: the question, what you found in the data,
what your model does and how well it does it, which features SHAP flagged, and — most importantly
— what you would *not* yet conclude, and what you would check next.

## Assessment focus

We are not looking for the highest accuracy. We are looking for a defensible pipeline, an honest
evaluation, and a biological interpretation that the group can stand behind and discuss.
