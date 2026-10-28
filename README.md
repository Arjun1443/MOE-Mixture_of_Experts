# Sparse Mixture of Experts — Study Notebook

[**Launch MoE Routing Lab →**](https://moe-mixtureofexperts.streamlit.app/) · [Deployment guide](DEPLOYMENT.md)


## Interactive Streamlit app

A companion application is available in [streamlit_app/app.py](streamlit_app/app.py).

```bash
python -m pip install -r streamlit_app/requirements.txt
python -m streamlit run streamlit_app/app.py
```

This is a seeded routing simulation, not trained language-model inference.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Community Cloud setup, validation coverage, and runtime requirements. Deployment URL: https://moe-mixtureofexperts.streamlit.app/


A PyTorch study notebook exploring expert modules, top-k routing, noisy gating, and integration into a decoder-style language model.

## Why this project matters

Sparse routing provides a practical way to understand how a model can select a subset of expert networks for each token.

## Technical focus

PyTorch · neural networks · sparse routing · transformer blocks

## Explore the implementation

Open [the notebook](<Mixture_of_Experts_(MoE)_from_Scratch.ipynb>) to inspect preprocessing, implementation, and saved outputs.

## Run locally

```bash
git clone https://github.com/Arjun1443/MOE-Mixture_of_Experts.git
cd MOE-Mixture_of_Experts
python -m venv .venv
# Activate .venv before continuing.
python -m pip install torch jupyter
python -m jupyter notebook
```

Select `Mixture_of_Experts_(MoE)_from_Scratch.ipynb` in Jupyter and review its configuration before executing cells in order.

## Reproducibility notes

The notebook contains an upstream reference to [AviSoori1x/makeMoE](https://github.com/AviSoori1x/makeMoE). Credit for the referenced implementation remains with its original authors. Review the notebook's dataset-download and training cells before execution. This repository is presented as learning material, without claiming original authorship of the upstream architecture or code.

Saved notebook outputs describe a prior execution; they are not an independently reproduced benchmark. Dependency versions are not locked, so compatibility may require adjustment.


## Author

[Naga Anweesh Reddy](https://github.com/Arjun1443/Naga-Anweesh-Reddy) · AI development and technical training
