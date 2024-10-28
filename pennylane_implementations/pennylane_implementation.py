import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pennylane as qml

def parse_hamiltonian_file(filename):
    hamil_info = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse the header line to extract n_wires
    header_parts = lines[0].strip().split()
    if len(header_parts) >= 3 and header_parts[-1].isdigit():
        n_wires = int(header_parts[-1])
    else:
        raise ValueError("Cannot parse the number of wires from the header line.")

    hamil_info["n_wires"] = n_wires
    hamil_list = []

    for line in lines[1:]:
        if not line.strip():
            continue  # Skip empty lines
        parts = line.strip().split()
        coeff = float(parts[0])
        operators = parts[1:]
        pauli_string = ['I'] * n_wires  # Initialize with Identity operators

        for op in operators:
            if len(op) < 2:
                continue  # Skip invalid entries
            pauli_char = op[0]
            qubit_idx = int(op[1:])
            pauli_string[qubit_idx] = pauli_char.upper()

        pauli_string = ''.join(pauli_string)
        hamil_list.append({"coeff": coeff, "pauli_string": pauli_string})

    hamil_info["hamil_list"] = hamil_list
    return hamil_info

class QVQEModel(torch.nn.Module):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = hamil_info["n_wires"]
        self.n_blocks = arch["n_blocks"]
        # Initialize parameters for RY gates
        self.params = torch.nn.ParameterList()
        for _ in range(self.n_blocks):
            layer_params = torch.nn.Parameter(torch.randn(self.n_wires))
            self.params.append(layer_params)
        
        # Define the device and qnode
        self.dev = qml.device('default.qubit', wires=self.n_wires)
    
        # Preprocess Hamiltonian
        self.hamiltonian = self.build_hamiltonian(hamil_info["hamil_list"])

        # Define the QNode with adjoint differentiation
        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def circuit(*params):
            for k in range(self.n_blocks):
                for i in range(self.n_wires):
                    qml.RY(params[k][i], wires=i)
                for i in range(0, self.n_wires - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, self.n_wires - 1, 2):
                    if i + 1 < self.n_wires:
                        qml.CNOT(wires=[i, i + 1])
            return qml.expval(self.hamiltonian)
    
        self.circuit = circuit

    def build_hamiltonian(self, hamil_list):
        coeffs = []
        ops = []
        for hamil in hamil_list:
            coeff = hamil["coeff"]
            pauli_string = hamil["pauli_string"]
            op_list = []
            for idx, op_char in enumerate(pauli_string):
                if op_char != 'I':
                    op = getattr(qml, f"Pauli{op_char}")(idx)
                    op_list.append(op)
            if op_list:
                if len(op_list) == 1:
                    ops.append(op_list[0])
                else:
                    ops.append(qml.operation.Tensor(*op_list))
            else:
                ops.append(qml.Identity(0))  # If all are identity, use Identity on wire 0
            coeffs.append(coeff)
        hamiltonian = qml.Hamiltonian(coeffs, ops)
        return hamiltonian

    def forward(self):
        return self.circuit(*self.params)

def train(model, optimizer, n_steps=1):
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        print(f"Expectation of energy: {loss.item()}")

def valid_test(model):
    with torch.no_grad():
        loss = model()
    print(f"Validation: expectation of energy: {loss.item()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="number of blocks, each contains one layer of "
        "RY gates and one layer of CNOT with "
        "ring connections",
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=10, help="number of training steps per epoch"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument(
        "--hamil_filename",
        type=str,
        default="/Users/rohitganti/Desktop/LatticeVQE/pennylane_implementations/hamiltonian.txt",
        help="filename of the Hamiltonian",
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hamil_info = parse_hamiltonian_file(args.hamil_filename)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = QVQEModel(arch={"n_blocks": args.n_blocks}, hamil_info=hamil_info)

    model.to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        # Train
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        train(model, optimizer, n_steps=args.steps_per_epoch)
        scheduler.step()

    # Final validation
    valid_test(model)

if __name__ == "__main__":
    main()
