# ENN554
Code for ENN554 Design of Renewable Energy Production Systems

# Installation
This guide assumes that you have python installed on your machine (e.g. from https://www.python.org/downloads/). Since there is no PYPI package (yet?) the recommended procedure is to clone this repo to your machine and create an experimental install. The recommended install procedure is as follows. 

### Set up the virtual environment
Open up a terminal and create a virtual environment
```bash
python -m venv [foo]/enn554

```
where you'll need to replace ```[foo]``` with the location where you wish to store the virtual environment. 

### Clone the repo. 

The easiest way to do this is using **GitHub Desktop**. Download and install https://desktop.github.com/. Open the app and sign in with a GitHub account (free). You can then clone the repository by 

1. Click **File → Clone repository**
2. Select the **URL** tab
3. Paste the repository address: https://github.com/cholette/enn554
4. Choose where to save it on your computer.
5. Click **Clone**

Click **Show in Explorer** (Windows) or **Show in Finder** (Mac). You now have a normal folder on your computer containing the notebooks and code. Alternatively, if you prefer the terminal, you can simply navigate to the desired folder and execute
```bash
git clone https://github.com/cholette/enn554
```

### Install the code
To install the code, activate your environment by executing the following in a terminal:
```powershell
[foo]\enn554\Scripts\Activate.ps1
```
for Windows and
```bash
source [foo]/enn554/bin/activate
```
for macOS / Linux.

Now, in your terminal, navigate to inside the ```enn554``` directory that you just cloned and execute the following command
```bash
python -m pip install -e .[notebooks]
```
or 

```bash
python -m pip install -e .[all]
```
if you want the developer tools. This is only necessary if you plan on contributing to the repository.

### Create the ```data``` and ```ouptuts``` directories
In the main folder of the repo, create these folders. The contents of the ``data`` folder will need to be periodically updated with files from the Canvas site. 

### Opening Jupyter Notebooks
If all has gone well, you will be able to execute the following command
```bash
jupyter-notebook notebooks/lectures/2_energy_fundamentals.ipynb
```
which will open one of the notebooks that doesn't require data. If you can run this, your installation has succeeded. Other notebooks may not work yet because they need input files to be placed in ``data``, which we will do as the semester progresses. 

### Alternative ways of running notebooks
[VS Code](https://code.visualstudio.com/download) is a popular programming tool that [allows you to run notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). Also, If you don't have your own machine handy, the eResearch folks at QUT have a [JupyterLab](https://jupyter.eres.qut.edu.au/hub/home) environment that can be used to run the code. The steps are similar to the above.


# Code summary
This repository provides companion code for the unit and includes Python and MATLAB utilities for the tutorials and lecture content. The Python library currently includes:
- `sun.py` --- a class for computing sun position using standard equations. 
- `parabolic_trough.py` --- a class for thermal modelling of parabolic trough collectors. Includes a dynamic model that is a synthesis of state-of-the art models. 
- `wind.py` --- a class for basic analysis of wind speed data and the annual energy production of wind turbines using power curves
-`collector_geometry.py` --- a currently very incomplete set of methods for computing surface normals and view factors. 
- `utilties.py` --- a set of general use utilities (e.g. for reading in TMY files).

You will need to create data/ and outputs/ folders locally --- the former is to hold input files (e.g parameter files, TMY files) while the outputs/ folder is for dumping any code outputs that need to be saved (jsons, output spreadsheets).

# Acknowledgements
This repository was developed with support from the Heavy Industry Low-carbon Transition Cooporative Research Centre (HILT CRC) [Coursework Development Grant](https://hiltcrc.com.au/education-training/#short_course_development_grants).