# SENSE CDT (University of Leeds) triplet-trainer course

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/leifdenby/AIMSIR_convml_tt/HEAD) [![tests](https://github.com/leifdenby/AIMSIR_convml_tt/actions/workflows/tests.yml/badge.svg)](https://github.com/leifdenby/AIMSIR_convml_tt/actions/workflows/tests.yml)

This repository contains material to work with the neural network model used in
[L Denby
(2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085190)
and was created for SENSE CDT training at University of Leeds on 6th
March 2024 (material from previous years:
[2021](https://github.com/leifdenby/SENSE_convml_tt/tree/2021), 
[2022](https://github.com/leifdenby/SENSE_convml_tt/tree/2022) and 
[2023](https://github.com/leifdenby/SENSE_convml_tt/tree/2023))

Exercises are stored as jupyter notebooks in [notebooks/](notebooks/)

## Getting started

You can either run the exercises in your browser on
[mybinder](https://mybinder.org/v2/gh/leifdenby/AIMSIR_convml_tt/HEAD)
(nothing to install, but slower and your work isn't saved), or on your own
computer. For the latter you will need two things:

1) A copy of the exercises (the repository you're looking at right now!)

2) [uv](https://docs.astral.sh/uv/), which installs `convml-tt` and all the
   other packages needed for the exercises

If you are on Windows some more detailed notes are given
[here](README.windows.md).

### 1. Downloading the exercises

Choose a suitable parent directory (for example your desktop, `~/Desktop`)
and clone this repository so that you have a local copy of the exercises

```bash
git clone https://github.com/leifdenby/AIMSIR_convml_tt
cd AIMSIR_convml_tt
```

In the execises you will work with a dataset and trained model that comes
bundled with `convml-tt` and instructions for how to download these is
contained within the exercises.

### 2. Install `convml-tt` and its dependencies with `uv`

First [install uv](https://docs.astral.sh/uv/getting-started/installation/),
for example on linux and macOS with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from inside the folder with the exercises, create an environment with
everything installed. Choose the pytorch build to use with an "extra": `cpu`
if you don't have an NVIDIA GPU (this is also the right choice on macOS, where
Apple Silicon GPUs are still used), otherwise the one matching your CUDA
version (`gpu-cu118`, `gpu-cu121` or `gpu-cu124`):

```bash
uv sync --extra cpu
```

## Exercises

From inside the folder with the exercises (e.g. `~/Desktop/AIMSIR_convml_tt`)
start up a jupyter session and get going with the exercises:

```bash
uv run jupyter notebook
```

The exercises are broken down as follows:

1) **Dimensionality reduction**: Examine how the neural
   network has used the embedding space; are all 100 dimensions necessary? Can
   we identify what features the neural network has learnt by comparing tiles
   in different parts of the embedding space? notebook: [1a_Use_PCA_analysis_to_study_tile_embeddings.ipynb](notebooks/1a_Use_PCA_analysis_to_study_tile_embeddings.ipynb)

2) **High-dimensional clustering**: use different
   clustering methods to study the extent to which the neural network has
   formed distinct clusters in the embedding space.
   notebook: [1b_Exploring_embedding_space_with_clustering_methods.ipynb](notebooks/1b_Exploring_embedding_space_with_clustering_methods.ipynb)

3) **Using your own input data**: either by generating synthetic input tiles or
   using your own data source you will work with the pre-trained model to study
   whether the trained neural network groups them together in the embedding
   space. notebook:
   [2_Working_with_your_own_data.ipynb](notebooks/2_Working_with_your_own_data.ipynb)
