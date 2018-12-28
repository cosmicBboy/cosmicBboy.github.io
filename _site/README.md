# README

This static blog site uses [jekyll](https://jekyllrb.com/) for the website
structure and style. It uses python and jupyter notebooks for content creation.

## Setup

Use conda to manage python virtualenv and dependencies. ([install link](https://conda.io/docs/user-guide/install/macos.html#install-macos-silent))

Create the env

```
conda create -n cosmicbboy-blog ipykernel python=3.6
pip install -r requirements.txt
```

Create jupyter kernel

```
source activate cosmicbboy-blog
# this will automatically create a kernel for this virtualenv
python -m ipykernel install --name cosmicbboy-blog
```
