## Quick Start
Clone the repo:
```
git clone https://github.com/LukeV37/ML_Tutorial_2025.git
```

## How To Run (3) Methods:
### (1) Locally on Unix Environment
Install dependencies:
`sudo apt install git git-lfs build-essential python3-venv curl`

Setup the virtual environment:
```
source setup.sh
```
Please be patient while pip installs packages and Pythia8 compiles...


> [!TIP]
> Run the jupyter notebooks in the `notebooks` directory. Otherwise, run the code directly from the terminal using python in the `scripts` directory.


### (2) Cloud Computing on Google Collab
Open the URL in your web browser and sign into your google account: [https://colab.research.google.com/github/LukeV37/ML_Tutorial_2025/blob/main/notebooks/GoogleColab.ipynb](https://colab.research.google.com/github/LukeV37/ML_Tutorial_2025/blob/main/notebooks/GoogleColab.ipynb)

> [!IMPORTANT]  
> To utilize GPU resources, click `Runtime` > `Change runtime type` > `T4 GPU`. Otherwise training will take longer on CPU.

### (3) Batch Servers
Run the code remotely on [UChicago Analysis Facility](https://af.uchicago.edu/jupyterlab/configure) or `ssh <username>@lxplus-gpu.cern.ch` 

## Misc
This tutorial demonstrates Top Tagging by performing binary classification on ttbar vs diboson W+jets events.
Particles are generated using pythia and the top 100 tracks - sorted by pT with features (pT, η, ϕ) - are input to neural networks.
Three architectures are demonstrated: Multilayer Perceptron, DeepSets, and Transformer Encoder.

> [!NOTE]
> With a large enough dataset, we can show that large transformer models are capable of learning topological structure of signal events.

> [!CAUTION]
> Using low level particle information, the models perform best with more events. Be careful training on less than 100k events!
