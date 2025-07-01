## Quick Start
Login to a machine with a gpu:
```
ssh <username>@login.af.uchicago.edu
```
```
ssh <username>@lxplus-gpu.cern.ch
```
Clone the repo:
```
git clone https://github.com/LukeV37/ML_Tutorial_2025.git
```
Enter the directory `cd ML_Tutorial_2025`.

## How To Run:
### (1) Unix Environment
First, setup the virtual environment directly from the terminal using:
```
source setup.sh
```
Please be patient while pip installs packages and Pythia8 compiles...

If running on UChicago, request a jupyter lab instance from [UChicago Analysis Facility](https://af.uchicago.edu/jupyterlab/configure).

If running on lxplus-gpu, start a jupyter server and use `jupyter notebook list` to open the notebook locally.

Open the jupyter notebook in the `notebooks` directory. Otherwise, run the code directly from the terminal using python in the `scripts` directory.

> [!TIP]  
> Ssh into the remote server and forward the port of your jupyter notebook server `ssh -L XXXX:localhost:XXXX <username>@lxplus.cern.ch`
> Then start a jupyter server using `jupyter notebook --no-browser --port=XXXX`
> Type `jupyter notebook list` and click the link to open the server locally in your browser.


### (2) Cloud Computing on Google Collab
No setup needed! Simply open the URL in your web browser and sign into your google account: [https://colab.research.google.com/github/LukeV37/ML_Tutorial_2025/blob/main/notebooks/GoogleColab.ipynb](https://colab.research.google.com/github/LukeV37/ML_Tutorial_2025/blob/main/notebooks/GoogleColab.ipynb)

> [!IMPORTANT]  
> To utilize GPU resources, click `Runtime` > `Change runtime type` > `T4 GPU`. Otherwise training will take longer on CPU.

## Misc
This tutorial demonstrates Top Tagging by performing binary classification on ttbar vs diboson W+jets events.
Particles are generated using pythia and the top 100 tracks - sorted by pT with features (pT, η, ϕ) - are input to neural networks.
Three architectures are demonstrated: Multilayer Perceptron, DeepSets, and Transformer Encoder.

> [!NOTE]
> With a large enough dataset, we can show that large transformer models are capable of learning topological structure of signal events.

> [!CAUTION]
> Using low level particle information, the models perform best with more events. Be careful training on less than 100k events!
