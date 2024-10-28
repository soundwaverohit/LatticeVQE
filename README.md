# Hamiltonian Description
The Hamiltonian we are using models a lattice gauge theory with SU(3) color degrees of freedom and fermionic hopping terms. The Hamiltonian has two main components: hopping terms and on-site mass terms.

This sentence uses `$` delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$


### Hopping Terms
The hopping terms describe fermions moving between neighboring lattice sites, mediated by an SU(3) gauge field:
$\[H_{\text{hop}} = -t \sum_{\langle i, j \rangle} \sum_{\alpha=1}^{3} \frac{\operatorname{Re} \operatorname{Tr} \left( U_{i,j} U_{i,j}^{\dagger} \right)}{2} \left( c_{i,\alpha}^{\dagger} c_{j,\alpha} + c_{j,\alpha}^{\dagger} c_{i,\alpha} \right)\]$

where \( t \) is the hopping strength, \( \langle i, j \rangle \) denotes neighboring sites, and \( U_{i,j} \) is an SU(3) matrix representing the gauge field on the link between sites \( i \) and \( j \). Here, \( c_{i, \alpha}^{\dagger} \) and \( c_{j, \alpha} \) are the fermionic creation and annihilation operators at site \( i \) and color \( \alpha \).

### Mass Terms
The on-site mass term describes the energy cost associated with occupying a site:

$ \[
H_{\text{mass}} = m \sum_{i=1}^{N} \sum_{\alpha=1}^{3} c_{i,\alpha}^{\dagger} c_{i,\alpha}
\] $
where \( m \) is the fermion mass, \( N \) is the total number of sites, and the sum over \( \alpha \) accounts for the three colors in SU(3).

### Total Hamiltonian
The complete Hamiltonian is the sum of the hopping and mass terms:
\[
H = H_{\text{hop}} + H_{\text{mass}}
\]

This Hamiltonian structure captures both the gauge field's influence on particle motion (via hopping terms) and the individual particle mass for each color on each site.
