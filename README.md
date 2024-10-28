# Hamiltonian Description

This implementation of a Variational Quantum Eigensolver (VQE) constructs a Hamiltonian with SU(3) symmetry on a 2D lattice, designed for use in a quantum lattice gauge theory simulation. The Hamiltonian includes hopping terms and on-site mass terms to model fermionic interactions in an SU(3) gauge theory.


## Parameters and Structure
- Lattice Parameters:
  g: Gauge coupling parameter.
  t: Hopping term coefficient.
  m: Mass term coefficient.
  lattice_size: The size of the lattice grid in 2D, defining the total number of lattice sites as num_sites = lattice_size * lattice_size.
  num_colors: Number of SU(3) colors, here set to 3.



### SU(3) Generators
The Hamiltonian uses SU(3) generators, represented by the Gell-Mann matrices, to define the gauge field. These matrices are:

G1 = [[0, 1, 0],
      [1, 0, 0],
      [0, 0, 0]]

G2 = [[0, -i, 0],
      [i, 0, 0],
      [0, 0, 0]]

G3 = [[1, 0, 0],
      [0, -1, 0],
      [0, 0, 0]]

