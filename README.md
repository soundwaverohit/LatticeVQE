# Simulating SU(3) LGT Fermionic Hamiltonian in a Simple 2D Lattice 

This repository contains implementations in torchquantum, tensorcircuit, and pennylane. The goal is to try out different ansatz's quantum circuit tensor networks and hybrid methods to approximate the correct ground state.

### Code Overview and Purpose

This code simulates a lattice gauge theory with SU(3) symmetry, targeting a 2D lattice model for simplicity. The goal is to study the quantum properties of a system with fermions interacting under an SU(3) gauge field, which mimics the interactions of particles in quantum chromodynamics. The code does this by:

1. **Constructing the Hamiltonian**: The Hamiltonian is built from the hopping, mass, and plaquette terms, each representing different types of interactions within the lattice gauge theory. This is in the SU(3)FermionicHamiltonian Directory with a further explanation

2. **Mapping the Hamiltonian to Qubits**: The fermionic Hamiltonian is mapped to a qubit operator using the Jordan-Wigner transformation, making it suitable for simulation on a quantum computer.

3. **Creating a Custom Ansatz Circuit**: A parameterized quantum circuit (ansatz) is designed for use in the Variational Quantum Eigensolver (VQE), an algorithm to approximate the ground state of the Hamiltonian.

4. **Running the VQE Algorithm**: The VQE algorithm optimizes the parameters of the ansatz circuit to minimize the expectation value of the Hamiltonian, approximating the minimum (or ground state) energy of the system.

5. **Including Plaquette Energy**: The code calculates the plaquette energy separately and adds it to the ground state energy obtained from the VQE.

---

### Significance of the Minimum Eigenvalue

In quantum mechanics, the **minimum eigenvalue of the Hamiltonian** represents the **ground state energy** of the system. This value is crucial because:

- It corresponds to the **lowest energy state** of the system, which is often the most stable state.
- Understanding the ground state properties provides insights into the **quantum phase** of the system, especially for complex systems like lattice gauge theories.
- For lattice gauge theories, the ground state energy also helps in studying **confinement** and **interaction dynamics** of particles under gauge fields, important for simulating and understanding systems like QCD.

In this context, finding the minimum eigenvalue is a primary objective because it reveals how the system behaves at its most fundamental level under the SU(3) gauge field interactions, which could provide insights into the nature of strong interactions in particle physics. 

This simulation process, specifically using VQE on quantum computers, is a step towards scalable quantum simulations of complex gauge theories, potentially enabling future studies of particle interactions beyond classical computational limits.
