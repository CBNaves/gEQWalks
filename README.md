# Generalized elephant quantum walks

Python code for simulating generalized Elephant Quantum Walks (gEQW). 

A generalized Elephant Quantum Walk is a discrete time quantum walk (DTQW) where a memory probability distribution is introduced to select the step sizes,
accordingly with the past time instants. The memory probability distribution used is the [q-Exponential](https://en.wikipedia.org/wiki/Q-exponential_distribution).
By changing the q parameter it is possible to simulate quantum walks ranging from the standard DTQW ( q = 0.5) to the elephant quantum walk ( q = infinity),
in which all the past time instants are equally probable to be selected. For more details see the references below.

* A Review on quantum walks - https://arxiv.org/pdf/1207.7283.pdf

* Elephant quantum qalk - https://doi.org/10.1103/PhysRevA.97.062112

* Generalized elephant quantum walk - https://www.nature.com/articles/s41598-019-55642-5

## Configuration and Usage

The extra packages needed for the simulation are only the **numpy**, **scipy** and **matplotlib**.

The gEQW.cfg file is where the simulations must be specified so that the program read it. For more details of 
the sintax used see the file. It's worth noting that only the **fermion** coin type is working for all 
simulation cases.

To run the code simply run the **main.py** file :)

## Simulation

Running the program will create a data directory where the simulation datas will be saved. The simulations data 
are separed by the dimension of the lattice and the directories are named accordingly with the date/time.

The files saved are

* **parameters.txt** - a copy of the parameters used in the simulation.

* **pd_t** - file where the joint position probability distribution of the coin in time = t is saved.

* **statistics.txt** - file where the mean position and variance for every time step is saved.
  See the file for more details.
                       
* **entanglement_entropy** (entang = True) - file where the von Neumann entropy, or entanglement entropy,
  of the coin system is saved in all the evolution.
  
* **trace_distance.txt** (trace_dist = True) - in cases where more than one coin is used it is possible to
calculate the trace distance between its states during the evolution.

For more details on the code implementation see the docstrings.
