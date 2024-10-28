Hamiltonian Description
This implementation of a Variational Quantum Eigensolver (VQE) constructs a Hamiltonian with SU(3) symmetry on a 2D lattice, designed for use in a quantum lattice gauge theory simulation. The Hamiltonian includes hopping terms and on-site mass terms to model fermionic interactions in an SU(3) gauge theory.


H_hopping = -t * ∑_{n, α} (1/2) * Re(Tr(U_{n, n+1} U_{n+1, n}†)) * c†_{n, α} c_{n+1, α}
