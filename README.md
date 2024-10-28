\section{Hamiltonian Description}

This implementation of a Variational Quantum Eigensolver (VQE) constructs a Hamiltonian with SU(3) symmetry on a 2D lattice, designed for use in a quantum lattice gauge theory simulation. The Hamiltonian includes hopping terms and on-site mass terms to model fermionic interactions in an SU(3) gauge theory.

\subsection{Parameters and Structure}

\begin{itemize}
    \item \textbf{Lattice Parameters:}
    \begin{itemize}
        \item \( g \): Gauge coupling parameter.
        \item \( t \): Hopping term coefficient.
        \item \( m \): Mass term coefficient.
        \item \texttt{lattice\_size}: The size of the lattice grid in 2D, defining the total number of lattice sites as \( \text{num\_sites} = \text{lattice\_size} \times \text{lattice\_size} \).
        \item \texttt{num\_colors}: Number of SU(3) colors, here set to 3.
    \end{itemize}
\end{itemize}

\subsection{SU(3) Generators}

The Hamiltonian uses SU(3) generators, represented by the Gell-Mann matrices, which define the gauge field. These matrices are:

\[
G_1 = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad
G_2 = \begin{pmatrix} 0 & -i & 0 \\ i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad
G_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}
\]

and five additional matrices (\(G_4\) through \(G_8\)), forming a complete set of Gell-Mann matrices.

\subsection{Hamiltonian Terms}

\subsubsection{Hopping Terms}

The hopping terms represent fermionic hopping across lattice sites and incorporate SU(3) gauge interactions. Each term is given by:

\[
H_{\text{hopping}} = -t \sum_{n, \alpha} \frac{1}{2} \text{Re} \left( \text{Tr}(U_{n, n+1} U_{n+1, n}^\dagger) \right) \hat{c}^\dagger_{n, \alpha} \hat{c}_{n+1, \alpha}
\]

where \( U \) is a randomly generated SU(3) matrix that represents the gauge field on each lattice link. The hopping terms are added in both the forward and reverse directions to maintain Hermiticity.

\subsubsection{On-Site Mass Terms}

The mass terms represent interactions at each lattice site for each color, with each term given by:

\[
H_{\text{mass}} = m \sum_{n, \alpha} \hat{c}^\dagger_{n, \alpha} \hat{c}_{n, \alpha}
\]

where \( \hat{c} \) and \( \hat{c}^\dagger \) are fermionic annihilation and creation operators, respectively, and \( m \) is the mass parameter.

\subsection{Fermionic Hamiltonian in Code}

The following Python code snippet initializes the hopping and on-site mass terms and constructs the fermionic Hamiltonian:

\begin{verbatim}
# Initialize hopping terms and on-site mass terms
fermionic_terms = {}

# Hopping terms for SU(3) fermionic hopping
for n in range(num_sites - 1):
    U = gauge_field[n]
    for alpha in range(num_colors):
        idx1 = flatten_index(n, alpha, num_colors)
        idx2 = flatten_index(n + 1, alpha, num_colors)
        coeff = -t * np.real(np.trace(U @ U.conj().T)) / 2
        fermionic_terms[f"+_{idx1} -_{idx2}"] = coeff
        fermionic_terms[f"+_{idx2} -_{idx1}"] = coeff

# On-site mass terms
for n in range(num_sites):
    for alpha in range(num_colors):
        idx = flatten_index(n, alpha, num_colors)
        fermionic_terms[f"+_{idx} -_{idx}"] = m

# Fermionic Hamiltonian
fermionic_hamiltonian = FermionicOp(fermionic_terms, num_spin_orbitals=num_sites * num_colors)
\end{verbatim}

\subsection{Qubit Mapping}

The Hamiltonian, initially in fermionic form, is mapped to a qubit representation using the Jordan-Wigner transformation:

\begin{verbatim}
# Map to qubit Hamiltonian
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
\end{verbatim}

This Hamiltonian forms the foundation for VQE optimization, allowing for energy minimization through circuit-based simulations of SU(3) interactions on a lattice.
