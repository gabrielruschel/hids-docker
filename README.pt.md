# hids-docker

Neste repositório se encontram as bases e o código utilizado nos experimentos para o artigo "Detecção de Anomalias: Um estudo voltado na Identificação de Ataques no Ambiente de Contêiner" - Gabriel Ruschel Castanhel, [Tiago Heinrich](https://github.com/h31nr1ch), [Fabrício Ceschin](https://github.com/fabriciojoc), [Carlos Maziero](http://www.inf.ufpr.br/maziero) - Publicado em SBSeg 2020 - WTICG

## Instalação

Para a realização dos testes é necessário o `Python 3.8.5`, também é recomendado o uso de um ambiente virtual.

### Clone

Faça o clone deste repositório para a sua estação de trabalho utilizando `git@github.com:gabrielruschel/hids-docker.git`

### Setup

Primeiramente é necessário instalar as dependências

```shell
$ pip install -r requirements.txt
```

## Execução

Para executar os testes, basta executar

```shell
$ python main.py <tamanho_janela>
```

Onde o argumento `tamanho_janela` especifica o tamanho da janela usada nos testes. 

É possível editar na seção main do código quais métodos específicos deseja testar, e também mudar a forma de construção das janelas (com todas as system calls ou com filtragem), alterando a função utilizada em `retrieve_dataset()` (`sliding_window_raw` para trabalhar com todas as system calls ou `sliding_window_filter` para realizar a filtragem).
