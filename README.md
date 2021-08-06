# HIDS-Docker #

In this repository we present a docker dataset based on system calls and the source code used for its experimental evaluation, more details are presented in [HIDS Docker Information] (you can also read the same README in [portuguese](README.pt.md)):

## Table of Contents ##
- [HIDS Docker Information](#hids-docker-information)
    - [ISCC-2021](#ISCC-2021)
    - [ERRC-2020](#ERRC-2020)
    - [SBSeg-2020](#SBSeg-2020)
- [How do I get set up?](#how-do-i-get-set-up?)
    - [Install](#install)
    - [Setup](#setup)
- [Examples](#examples)
- [Contribution guidelines](#contribution-guidelines)
- [License](#license)

## HIDS Docker Information ##

This study is ongoing and has already had results presented at some conferences.

### ISCC-2021 ###

Published in  IEEE ISCC 2021 : 26th IEEE Symposium on Computers and Communications (ISCC):
```
@inproceedings{paper3,
    author={Gabriel Ruschel Castanhel and Tiago Heinrich and Fabrício Ceschin and Carlos A. Maziero},
    title={Taking a Peek: An Evaluation of Anomaly Detection Using System calls for Containers},
    year={2021},
    series={26th IEEE Symposium on Computers and Communications (ISCC 2021)}
}
```
The paper could be found [here]().

### ERRC-2020 ###

Published in Regional Workshop on Information Security and Computer Systems (WRSeg) in the XVIII Regional School of Computer Networks (ERRC 2020):
```
@inproceedings{paper2,
    author={Gabriel Ruschel Castanhel and Tiago Heinrich and Fabrício Ceschin and Carlos A. Maziero},
    title={Sliding Window: The Impact of Trace Size in Anomaly Detection System for Containers Through Machine Learning},
    year={2020},
    series={Regional Workshop on Information Security and Computer Systems (WRSeg - ERRC)}
}
```

The paper could be found [here](https://www.researchgate.net/publication/346607168_Sliding_Window_The_Impact_of_Trace_Size_in_Anomaly_Detection_System_for_Containers_Through_Machine_Learning).

### SBSeg-2020 ###

Published in Brazilian Symposium on Information and Computational Systems Security (SBSeg) 2020 - XIV Workshop of Scientific Initiation and Graduation Works (WTICG):
```
@inproceedings{paper1,
    author={Gabriel Ruschel Castanhel and Tiago Heinrich and Fabrício Ceschin and Carlos A. Maziero},
    title={Detecção de Anomalias: Estudo de Técnicas de Identificação de Ataques em um Ambiente de Contêiner},
    year={2020},
    series={Undergraduate Research Workshop - Brazilian Security Symposium (WTICG - SBSeg)}
}
```

The paper could be found [here](https://www.researchgate.net/publication/346246313_Deteccao_de_Anomalias_Estudo_de_Tecnicas_de_Identificacao_de_Ataques_em_um_Ambiente_de_Conteiner).

## How do I get set up? ##

The following components should be installed:

```
* python 3.8.5
* pip3 20.0.2
  * scikit-learn 0.22.2
```

### Install ###

Python3 and pip3 are required for testing. Arch linux installation guide:
```
sudo pacman -S python python-pip
```

Debian installation guide:
```
sudo apt install python3 python3-pip
```

### Setup ###

Clone this repository:
```
git clone https://github.com/gabrielruschel/hids-docker
```

And install the dependencies using pip3:
```
cd hids-docker/
pip3 install -r requirements.txt
```

## Examples ##

To run the tests, just use the following command:
```
python3 main.py [-d {sbseg,iscc}] [-f {raw,filter}] <window_size>
```
* `-d`: specifies which dataset to use (default: iscc)
* `-f`: specifies which filter mode to use (default: raw)
* `window_size`: specifies the size of the window be used in the tests.

The `-h` argument shows the help message. It is possible to edit in the main section of the code which specific methods you want to test.

## Contribution guidelines ##

* [gabrielruschel](https://github.com/gabrielruschel) (Gabriel Ruschel Castanhel) (owner)
* [h31nr1ch](https://github.com/h31nr1ch) (Tiago Heinrich) (contributor)
* [fabriciojoc](https://github.com/fabriciojoc) (Fabrício Ceschin) (contributor)
* [cmaziero](https://github.com/cmaziero) (Carlos Maziero) (contributor)
