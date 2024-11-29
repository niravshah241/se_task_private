## Finite element calculations using FEniCSx and Introduction to scientific machine learning using PyTorch

### Finite element calculations using FEniCSx

We consider domain $\Omega \subset \mathbb{R}^3$ with boundary $\partial \Omega = \Gamma_D \cup \Gamma_N$. Here, $n$ is the outward pointing normal. The poisson problem is given as:

$$ -\Delta u = f \ \text{in} \ \Omega \, $$
$$ u = 0 \ \text{on} \ \Gamma_D \ ,$$
$$ \nabla u \cdot n = 0 \ \text{on} \ \Gamma_N \ .$$

Corresponding weak form is given as: find $u \in \mathbb{V}$ such that:

$$ a(u, v) = L(v) \ \forall \ v \in \mathbb{V} \ , $$
$$ a(u, v) = \int_{\Omega} \nabla u \cdot \nabla v dx \ , $$
$$ L(v) = \int_{\Omega} fv dx \ , $$

where, $\mathbb{V}$ is  a suitable function space, $a(\cdot, \cdot)$ is the bilinear form and $L(\cdot)$ is the linear form.

In this example we consider,

* $\Omega = [0, 1] \times [0, 1] \times [0, 1]$ (Unit cube)  
* $\Gamma_D = \lbrace (1, y, z) \cup (x, 1, z) \cup (x, y, 1) \rbrace$  
* $\Gamma_N = \lbrace (0, y, z) \cup (x, 0, z) \cup (x, y, 0) \rbrace$  
* $f = 10  e^{((x - 0.40)^2 + (y - 0.55)^2 + (z - 0.27)^2)}$  

The implementation is given in the folder fenicsx. To run the C++ demo (num_procs = number of processes):

```
cd fenicsx
mkdir build_folder
cd build_folder
ffcx ../poisson.py
cmake ..
make
mpirun -np num_procs ./demo_poisson
```

To run the python demo (num_procs = number of processes):

```
mpirun -np num_procs python3 fenicsx_python.py
```

Also, see the section Dependencies below.

### Introduction to scientific machine learning using PyTorch

This part focuses on developing a python package ```sciml_modules```. We consider a problem of approximating the function between input ($x$) and output ($y$) of the form:

$$y = sin(W_k x + b_k)$$  

where, $W_k$ and $b_k$ are the known weights and biases. This relationship can be represented using an input-output artificial neural network (no hidden layers). We randomly generate $x, W_k, b_k$ and calculate the output $y$. After generating, the input-output data $x, y$, we train a neural network to apprxoimate the known weights - biases and compare the absolute errors.

The package ```sciml_modules``` is divided into modules: Dataset, Neural net, Train-Validate (for serial training), additionally, Partitioned dataset and Train-Validate-Distributed (for data-parallel distributed training for small datasets) and Wrappers (some additional helper functions). The unit tests are given in ```test_unit.py``` (serial) and ```test_unit_parallel.py``` (distributed) in the folder ```pytorch/sciml_modules```.

### Dependencies

- [DOLFINx](https://fenicsproject.org/)
- [PyTorch](https://pytorch.org/)

It is recommended to use the docker images and provided ```docker-compose.yaml``` file. This will ensure proper library versions are used.

```docker compose -f docker-compose.yaml up ```

It will create two containers for FEniCSx and PyTorch. Alternatively, the individual ```Dockerfile```s are provided in the folders ```fenics_docker``` and ```pytorch_docker```. Afterwards, change directory to the cloned github repository and follow the steps above.
