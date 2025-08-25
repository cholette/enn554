# enn554
Code for ENN554 Design of Renewable Energy Production Systems

# Summary
This repository provides companion code for the unit and includes Python and MATLAB utilities for the tutorials and lecture content. The Python library currently includes:
- `sun.py` --- a class for computing sun position using standard equations. 
- `parabolic_trough.py` --- a class for thermal modelling of parabolic trough collectors. Includes a dynamic model that is a synthesis of state-of-the art models. 
- `wind.py` --- a class for basic analysis of wind speed data and the annual energy production of wind turbines using power curves. 
-`collector_geometry.py` --- a currently very incomplete set of methods for computing surface normals and view factors. 
- `utilties.py` --- a set of general use utilities (e.g. for reading in TMY files).

You will need to create data/ and outputs/ folders locally --- the former is to hold input files (e.g parameter files, TMY files) while the outputs/ folder is for dumping any code outputs that need to be saved (jsons, output spreadsheets).

# Acknowledgements
This repository was developed with support from the Heavy Industry Low-carbon Transition Cooporative Research Centre (HILT CRC) [Coursework Development Grant](https://hiltcrc.com.au/education-training/#short_course_development_grants).