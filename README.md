ML-Algo: Hyperbolic-ANN + Kernel Ensemble + CatBoost

Quick start
- Install: `pip install -e .` (optionally `pip install catboost faiss-cpu`)
- Run tests: `pytest -q`
- Quick CPU training: `python scripts/train_local.py --nrows 2000 --iterations 300`

Train on Colab (GPU)
- Open the notebook and run all cells:
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Afeks214/ML-Algo/blob/main/notebooks/colab_training.ipynb)

Docs
- Runbook (GPU): docs/runbooks/colab_gpu.md
- API contracts: docs/api_contracts.md
- Reproducibility: docs/runbooks/reproducibility.md

