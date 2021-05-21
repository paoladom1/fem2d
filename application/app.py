import sys
import numpy as np

from .utils import (
    read_mesh,
    create_local_K,
    create_local_b,
    assembly,
    apply_conditions,
    calculate_fem,
)


def run():
    """Backbone of FEM1D process"""
    args = sys.argv[1:]
    # expects 1 argument with the path to the file with gig dat file
    data_filename = args[0]

    # fill the mesh data
    mesh = read_mesh(data_filename)

    # get info read from mesh
    k, Q = mesh.parameters
    elements = mesh.elements
    nodes = mesh.nodes

    # create local systems
    local_K_array = np.array([create_local_K(element, k) for element in elements])
    local_b_array = np.array([create_local_b(element, Q) for element in elements])

    # build base K and b from assembly
    K, b = assembly(nodes, elements, local_K_array, local_b_array)

    # apply neumann and dirichlet conditions
    K, b = apply_conditions(mesh.neumann_conditions, mesh.dirichlet_conditions, K, b)

    T = calculate_fem(K, b)
    print("T: {0}".format(T))
