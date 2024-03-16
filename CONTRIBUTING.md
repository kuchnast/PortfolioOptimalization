# Contributing guide
### Main informations if you would like to help in this project

### 1. Fork
> git clone **<FORK_GIT_REPO_PATH>**

### 2. Preparing local workspace
Make sure you have instaled python in version 3.10 or higher by typing:
> python --version

To start developing first download 'virtualenv' and create new virtual environment.
> pip install virtualenv
> 
> python -m virtualenv venv

Now activate it.
### Linux
> source venv/bin/activate
### Windows
>.\venv\Scripts\activate.bat

### What next
Install development packages.
> pip install -r dev-requirements.txt

In order to run pytest tests:
> pytest

In order to run formatting and syntax tests **(not necessary)**:
> tox -e black-check,mypy,flake8

To run script just type command:
> script

Before submitting automatically format source files:
>tox -e black
