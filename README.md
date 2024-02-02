# Supporting material

This repository contains the supporting material and codes to reproduce most of the results presented in my thesis.

## Pre-requisites

### Firedrake installation

It is mandatory to install [Firedrake](https://www.firedrakeproject.org/) beforehand, [see here](https://www.firedrakeproject.org/download.html) for download instructions. 
However, to use the same Firedrake (and its dependencies) version as used in my original work, please download the Firedrake install script and run it instead of the usual instructions
in the Firedrake website:

```shell
firedrake-install --doi 10.5281/zenodo.10605321
```

Please be careful regarding the output messages to ensure that the installation was successful.

### Additional dependencies

After the Firedrake installation, please activate the Firedrake virtual env. Please change the following command according to the Path you have installed the Firedrake in your computer:

```shell
source /path/to/firedrake/bin/activate
```

After that, install the additional dependencies required to run all the codes:

```shell
pip install -r requirements.txt
```

## Running the numerical experiments

Almost all of the numerical experiments are available in this repository in the directory [script](https://github.com/volpatto/supporting_material_phd_thesis/tree/main/scripts). Be aware that you must activate the Firedrake virtual env before the execution of any Python script (or notebooks):

```shell
source /path/to/firedrake/bin/activate
```

If you are using VS Code or PyCharm (or any IDE of your preference), please configure it to use the Firedrake virtual env correctly.

For convenience, Python Notebooks are provided in the directory [post-processing](https://github.com/volpatto/supporting_material_phd_thesis/tree/main/post-processing) to plot some of the results published in the thesis.

## Contact

Please feel free to contact me if you have interest in something, questions, etc. You can open an issue in this repository as well.
