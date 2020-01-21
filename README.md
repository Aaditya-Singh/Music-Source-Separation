# Music Source Separation using PLCA and Deep Learning.

In this project, we tried to separate the different sources of sound in a music piece such as bass, drums, vocals etc. from each other using Probabilistic Latent Component Analysis (PLCA) and Convolutional Neural Networks (CNN). We used Bach-10 dataset for source separation and MIREX-05 dataset for fundamental frequency extraction.

UGP.ipynb contains the code for Source-Filter based Probabilistic Latent Component Analysis.
DL.ipynb contains the code for Deep Probabilistic Framework-I.
ML.ipynb contains the code for Deep Probabilistic Framework-II.
UGP.py, DL.py and ML.py are the python scripts for running UGP.ipynb, DL.ipynb and ML.ipynb on GPU.
FE.py contains the code for the fundamental frequency estimation on GPU. 

The codes are written in Python-3 using TensorFlow 2.0 Framework.
