If you'd like to train and play with the models, you can either download the dataset directly or generate them manually. It might take a while to generate all the data so we recommend you download them directly. 

## Download Dataset
The data are hosted on Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4122270.svg)](https://doi.org/10.5281/zenodo.4122270)

You can download all the files in the terminal
```bash
# this will download pendulum-gym-image-dataset-train.pkl (210.5MB)
wget https://zenodo.org/record/4122270/files/pendulum-gym-image-dataset-train.pkl
# this will download pendulum-gym-image-dataset-test.pkl (210.5MB)
wget https://zenodo.org/record/4122270/files/pendulum-gym-image-dataset-test.pkl
# this will download cartpole-gym-image-dataset-rgb-u9-train.pkl (4.5GB)
wget https://zenodo.org/record/4122270/files/cartpole-gym-image-dataset-rgb-u9-train.pkl
# this will download cartpole-gym-image-dataset-rgb-u9-test.pkl (4.5GB)
wget https://zenodo.org/record/4122270/files/cartpole-gym-image-dataset-rgb-u9-test.pkl
# this will download acrobot-gym-image-dataset-rgb-u9-train.pkl (4.5GB)
wget https://zenodo.org/record/4122270/files/acrobot-gym-image-dataset-rgb-u9-train.pkl
# this will download acrobot-gym-image-dataset-rgb-u9-test.pkl (4.5GB)
wget https://zenodo.org/record/4122270/files/acrobot-gym-image-dataset-rgb-u9-test.pkl

# please only download the following if you would like to train
# baseline HGN on the pendulum task
# this will download pendulum-gym-image-dataset.pkl (421.1MB)
wget https://zenodo.org/record/4122270/files/pendulum-gym-image-dataset.pkl
```

Plase download or symlink all the data to this folder so that the trainers can find them. 

## Generate Dataset Manually
### Pendulum
Running
```
python pend_dataset.py
```
will generate
- `pendulum-gym-image-dataset.pkl`
which will then be splited into 
- `pendulum-gym-image-dataset-train.pkl`
- `pendulum-gym-image-dataset-test.pkl`

If you would like to train baseline Hamiltonian Generative Network on the pendulum task, please don't delete the first file (`pendulum-gym-image-dataset.pkl`) since in our experiments, using only `pendulum-gym-image-dataset-train.pkl` to train HGN couldn't let it learn meaningful reconstruction images.

### CartPole
Running
```
python cart_dataset.py
```
will generate
- `cartpole-gym-image-dataset-rgb-u9.pkl`
which will then be splited into 
- `cartpole-gym-image-dataset-rgb-u9-train.pkl`
- `cartpole-gym-image-dataset-rgb-u9-test.pkl`

You can safely delete the first file since it is not used in the code and all the infomation can be retrieved from the last two files. 

### Acrobot
Running
```
python acro_dataset.py
```
will generate
- `cartpole-gym-image-dataset-rgb-u9.pkl`
which will then be splited into 
- `cartpole-gym-image-dataset-rgb-u9-train.pkl`
- `cartpole-gym-image-dataset-rgb-u9-test.pkl`

You can safely delete the first file since it is not used in the code and all the infomation can be retrieved from the last two files. 
