# Hamiltonian Description
The Hamiltonian we are using models a lattice gauge theory with SU(3) color degrees of freedom and fermionic hopping terms. The Hamiltonian has two main components: hopping terms and on-site mass terms.

### Hopping Terms
The hopping terms describe fermions moving between neighboring lattice sites, mediated by an SU(3) gauge field:
```math
H_{\text{hop}} = -t \sum_{\langle i, j \rangle} \sum_{\alpha=1}^{3} \frac{\text{Re} ( \text{Tr} ( U_{i,j} U_{i,j}^{\dagger} ))}{2} ( c_{i,\alpha}^{\dagger} c_{j,\alpha} + c_{j,\alpha}^{\dagger} c_{i,\alpha} )
```

where \( t \) is the hopping strength, \( \langle i, j \rangle \) denotes neighboring sites, and \( U_{i,j} \) is an SU(3) matrix representing the gauge field on the link between sites \( i \) and \( j \). Here, \( c_{i, \alpha}^{\dagger} \) and \( c_{j, \alpha} \) are the fermionic creation and annihilation operators at site \( i \) and color \( \alpha \).

### Mass Terms
The on-site mass term describes the energy cost associated with occupying a site:
```math
H_{\text{mass}} = m \sum_{i=1}^{N} \sum_{\alpha=1}^{3} c_{i,\alpha}^{\dagger} c_{i,\alpha}
```


where \( m \) is the fermion mass, \( N \) is the total number of sites, and the sum over \( \alpha \) accounts for the three colors in SU(3).

### Plaquette Terms: 
The plaquette terms describes the 
```math
H_{\text{plaq}} = \frac{g^2}{2} \sum_{\square} \text{Re} \, \text{Tr} \left( U_{ij} U_{jk} U_{kl} U_{li} \right)
```
The plaquette term in the Hamiltonian represents the magnetic interaction between the gauge fields on a lattice, specifically in lattice gauge theory. It encapsulates the non-Abelian flux through a square (plaquette) of the lattice, which corresponds to the interaction between gauge fields in a closed loop.

### Total Hamiltonian
The complete Hamiltonian is the sum of the hopping and mass terms:
```math
H = H_{\text{hop}} + H_{\text{mass}}
```

This Hamiltonian structure captures both the gauge field's influence on particle motion (via hopping terms) and the individual particle mass for each color on each site.
