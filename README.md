# HIDS-Docker #

In this repository we present a docker dataset based on system calls and the source code used for its experimental evaluation, more details are presented in [HIDS Docker Information] (you can also read the same README in [portuguese](README.pt.md)):

## Table of Contents ##
- [HIDS Docker Information](#hids-docker-information)
- [How do I get set up?](#how-do-i-get-set-up?)
- [Install](#install)
- [Setup](#setup)
- [Examples](#examples)
- [Contribution guidelines](#contribution-guidelines)
- [License](#license)

## HIDS Docker Information ##

Published in Brazilian Symposium on Information and Computational Systems Security (SBSeg) 2020 - XIV WORKSHOP OF SCIENTIFIC INITIATION AND GRADUATION WORKS (WTICG):
```
@inproceedings{paper1,
    author={Gabriel Ruschel Castanhel and Tiago Heinrich and Fabrício Ceschin and Carlos A. Maziero},
    title={Detecção de Anomalias: Estudo de Técnicas de Identificação de Ataques em um Ambiente de Contêiner},
    year={2020},
    series={Undergraduate Research Workshop - Brazilian Security Symposium (WTICG - SBSeg)}
}
```
Paper submitted to Regional Workshop on Information Security and Computer Systems (WRSeg) in the XVIII Regional School of Computer Networks (ERRC 2020):
```
@inproceedings{paper2,
    author={Gabriel Ruschel Castanhel and Tiago Heinrich and Fabrício Ceschin and Carlos A. Maziero},
    title={Sliding Window: The Impact of Trace Size in Anomaly Detection System for Containers Through Machine Learning},
    year={2020},
    series={Submitted to Regional Workshop on Information Security and Computer Systems (WRSeg - ERRC)}
}
```

## How do I get set up? ##
* python 3.8.5
* pip3 20.0.2
```
* scikit-learn 0.22.2
```

## Install ##

Python3 and pip3 are required for testing. Arch linux installation guide:
```
sudo pacman -S python python-pip
```
Debian installation guide:
```
sudo apt install python3 python3-pip
```

### Setup

Clone this repository

```
git clone https://github.com/gabrielruschel/hids-docker
```

And install the dependencies using pip3

```
cd hids-docker/
pip3 install -r requirements.txt
```

## Examples

To run the tests, just use the following command

```
python3 main.py <tamanho_janela>
```

Where the `tamanho_janela` (`window_size`) argument specifies the size of the window used in the tests.

It is possible to edit in the main section of the code which specific methods you want to test, and also change the form of construction of the windows (with all system calls or with filtering), changing the function used in `retrieve_dataset ()` (`sliding_window_raw` to work with all system calls or `sliding_window_filter` to perform the filtering).

## Contribution guidelines ##
* gabrielruschel (Gabriel Ruschel Castanhel) (owner)
* h31nr1ch (Tiago Heinrich) (contributor)
* fabriciojoc (Fabrício Ceschin) (contributor)
* maziero (Carlos Maziero) (contributor)
