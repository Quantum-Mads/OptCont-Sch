import torch
import numpy as np
from config_manager import DeviceManager, DatetypeManager
"""
This file stores the matrix representations of some commonly used single-qubit gates.
"""

# Ensure gates are created on GPU or CPU
device_manager = DeviceManager()
device = device_manager.get_device()
# Set the datetype
datetype_manager = DatetypeManager()
datetype = datetype_manager.get_datetype()

def rx(parameter):
    parameter = parameter / 2
    return torch.tensor([[np.cos(parameter), -1j*np.sin(parameter)],
                         [-1j*np.sin(parameter), np.cos(parameter)]],
                        dtype=datetype, device=device)
def ry(parameter):
    parameter = parameter / 2
    return torch.tensor([[np.cos(parameter), -np.sin(parameter)],
                         [np.sin(parameter), np.cos(parameter)]],
                        dtype=datetype, device=device)
def rz(parameter):
    parameter = parameter / 2
    return torch.tensor([[np.exp(-1j*parameter), 0],
                         [0, np.exp(1j*parameter)]],
                        dtype=datetype, device=device)
def phase_gate(parameter):
    return torch.tensor([[1, 0],
                         [0, np.exp(1j*parameter)]],
                        dtype=datetype, device=device)
def global_phase(parameter):
    return torch.tensor([[np.exp(1j*parameter), 0],
                         [0, np.exp(1j*parameter)]],
                        dtype=datetype, device=device)
def u(parameters):
    return torch.tensor([[np.cos(parameters[0]/2), -np.exp(1j*parameters[2])*np.sin(parameters[0]/2)],
                         [np.exp(1j*parameters[1])*np.sin(parameters[0]/2), np.exp(1j*(parameters[1]+parameters[2]))*np.cos(parameters[0]/2)],],
                        dtype=datetype, device=device)
gate_set = {
    "x": torch.tensor([[0, 1], [1, 0]], dtype=datetype, device=device),
    "y": torch.tensor([[0, -1j], [1j, 0]], dtype=datetype, device=device),
    "z": torch.tensor([[1, 0], [0, -1]], dtype=datetype, device=device),
    "-z": torch.tensor([[-1, 0], [0, 1]], dtype=datetype, device=device),
    "s": torch.tensor([[1, 0], [0, 1j]], dtype=datetype, device=device),
    "s_dagger": torch.tensor([[1, 0], [0, -1j]], dtype=datetype, device=device),
    "h": torch.tensor([[1, 1], [1, -1]] , dtype=datetype, device=device) / np.sqrt(2),
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "phase_gate": phase_gate,
    "global_phase": global_phase,
    "u": u
}
gate_names = ['x', 'y', 'z', '-z', 's', 's_dagger', 'h', 'rx', 'ry', 'rz', 'phase_gate', 'global_phase', 'u']