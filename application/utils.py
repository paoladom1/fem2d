import numpy as np

from math import sqrt
from functools import reduce
from .classes import Node, Element, Mesh, Condition


def find_node_in_nodes(node_id, nodes):
    """Finds a node by id in list of nodes

    :node_id: id of node to find
    :nodes: list of nodes
    :returns: node::<Node>

    """
    return next((node for node in nodes if node.id == node_id))


def read_mesh(filename):
    """Reads data from file <filename> to fill the mesh

    :filename: name of the file to be read

    :returns: Mesh

    """
    with open(f"application/data/{filename}.dat") as f_in:
        # creates array of lines without empty lines
        lines = list(
            map(
                lambda line: line.split(" "),
                filter(None, (line.rstrip() for line in f_in)),
            )
        )
        # removes spaces/tabs
        lines = [list(filter(lambda x: x != "", line)) for line in lines]

        # current line read
        current_line = 0

        # read k and Q
        k, Q = map(float, lines[current_line])
        current_line += 1  # go to nodes/elements data after reading k, Q

        # nodes, elements, cant_dirichlet, cant_neumann
        cant_nodes, cant_elements, cant_dirichlet, cant_neumann = map(int, lines[1])

        # jump parameters and start_coordinates lines
        current_line += 2

        nodes = []
        for line in lines[current_line : current_line + cant_nodes]:
            # parse line
            node_id, node_x, node_y = line

            # create node object
            node = Node(int(node_id), float(node_x), float(node_y))

            # append to node
            nodes.append(node)

            # nextline
            current_line += 1  # jump each node line

        # end_coordinates + start_elements
        current_line += 2
        elements = []
        for line in lines[current_line : current_line + cant_elements]:
            # parse line
            element_id, element_node1_id, element_node2_id, element_node3_id = map(
                int, line
            )

            # get nodes objects
            element_node1 = find_node_in_nodes(element_node1_id, nodes)
            element_node2 = find_node_in_nodes(element_node2_id, nodes)
            element_node3 = find_node_in_nodes(element_node3_id, nodes)

            # create element object
            element = Element(element_id, element_node1, element_node2, element_node3)

            # append to elements list
            elements.append(element)

            # nextline
            current_line += 1

        # skip end_elements and start_dirichlet
        current_line += 2

        dirichlet_conditions = []
        for line in lines[current_line : current_line + cant_dirichlet]:
            # parse line
            condition_node_id, condition_value = line

            # find node in nodes
            condition_node = find_node_in_nodes(int(condition_node_id), nodes)

            # create condition object
            condition = Condition(condition_node, float(condition_value))

            # append condition to list
            dirichlet_conditions.append(condition)

            # nextline
            current_line += 1

        # skip end dirichlet and start_neumann
        current_line += 2

        neumann_conditions = []
        for line in lines[current_line : current_line + cant_neumann]:
            # parse line
            condition_node_id, condition_value = line

            # find node in nodes
            condition_node = find_node_in_nodes(int(condition_node_id), nodes)

            # create condition object
            condition = Condition(condition_node, float(condition_value))

            # append condition to list
            neumann_conditions.append(condition)

    return Mesh([k, Q], nodes, elements, dirichlet_conditions, neumann_conditions)


def create_local_K(element, k):
    """Creates the matrix of local K for FEM2D

    :element: the local element
    :k: the value of thermal conductivity

    :returns: matrix for current element

    """
    # get nodes from element
    node1 = element.node1
    node2 = element.node2
    node3 = element.node3

    # calculate D
    D = np.linalg.det(
        np.matrix(
            [
                [(node2.x - node1.x), (node2.y - node1.y)],
                [(node3.x - node1.x), (node3.y - node1.y)],
            ]
        )
    )

    # calculate Ae with Heron formula
    a = np.linalg.norm([node2.x - node1.x, node2.y - node1.y])
    b = np.linalg.norm([node3.x - node2.x, node3.y - node2.y])
    c = np.linalg.norm([node3.x - node1.x, node3.y - node1.y])
    s = (a + b + c) / 2

    Ae = sqrt(s * (s - a) * (s - b) * (s - c))

    # create matrix A
    A = np.matrix(
        [[node3.y - node1.y, node1.y - node2.y], [node1.x - node3.x, node2.x - node1.x]]
    )

    # create matrix B
    B = np.matrix([[-1, 1, 0], [-1, 0, 1]])

    # create transpose matrixes
    At = A.transpose()
    Bt = B.transpose()

    return (k * Ae / (D * D)) * Bt @ At @ A @ B


def create_local_b(element, Q):
    """Creates the vector of local b for FEM2D

    :element: the local element
    :Q: the value heat source

    :returns: [bi, bi, bi]

    """
    # get nodes from element
    node1 = element.node1
    node2 = element.node2
    node3 = element.node3

    # calculate J
    J = np.linalg.det(
        np.matrix(
            [
                [node2.x - node1.x, node3.x - node1.x],
                [node2.y - node1.y, node3.y - node1.y],
            ]
        )
    )

    # calculate bi
    bi = Q * J / 6

    return np.repeat(bi, 3)


def assembly(nodes, elements, localKs, localbs):
    """Assembly K and b

    :nodes: the list of nodes
    :elements: the list of elements
    :localKs: the array of local Ks
    :localBs: the array of local bs

    :returns: K, b

    """
    # get length of nodes and elements lists
    num_nodes = len(nodes)
    num_elements = len(elements)

    # initialize K and b as zeroes
    K = np.zeros((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    for index, element in enumerate(elements):
        # get nodes from element
        node1 = element.node1
        node2 = element.node2
        node3 = element.node3

        # fill K
        localK = localKs[index]
        # row 1
        K[node1.index][node1.index] += localK[0][0]
        K[node1.index][node2.index] += localK[0][1]
        K[node1.index][node3.index] += localK[0][2]
        # row 2
        K[node2.index][node1.index] += localK[1][0]
        K[node2.index][node2.index] += localK[1][1]
        K[node2.index][node3.index] += localK[1][2]
        # row 3
        K[node3.index][node1.index] += localK[2][0]
        K[node3.index][node2.index] += localK[2][1]
        K[node3.index][node3.index] += localK[2][2]

        # fill b
        local_b = localbs[index]
        b[node1.index] += local_b[0]
        b[node2.index] += local_b[1]
        b[node3.index] += local_b[2]

    return K, b


def apply_conditions(neumann_conditions, dirichlet_conditions, K, b):
    """Apply neumann and dirichlet conditions to K and b

    :neumann_conditions: the neumann_conditions to be applied
    :dirichlet_condition: the dirichlet_conditions to be applied
    :K: the global matrix K
    :b: the global vector b

    :returns: K, b

    """
    # temp variables
    temp_K = K
    temp_b = b

    # applying neumann
    for neumann_condition in neumann_conditions:
        neumann_node = neumann_condition.node
        neumann_value = neumann_condition.value
        temp_b[neumann_node.index] += neumann_value

    # applying dirichlet
    # sorting list of dirichlet conditions by index
    sorted_conditions = sorted(
        dirichlet_conditions, key=lambda condition: condition.node.id
    )

    # use i to substract that amount of indexes to the original index
    for i, dirichlet_condition in enumerate(sorted_conditions):
        dirichlet_node = dirichlet_condition.node
        dirichlet_value = dirichlet_condition.value
        # delete from K, the object (index) from axis 0 (row)
        temp_K = np.delete(temp_K, dirichlet_node.index - i, axis=0)
        # delete from b, the object (index)
        temp_b = np.delete(temp_b, dirichlet_node.index - i)

        # pass value from column in K to b converted
        for index, row in enumerate(temp_K):
            cell = row[dirichlet_node.index - i]
            temp_b[index] += -1 * dirichlet_value * cell

        # delete from K, the object (index) from axis 1 (column)
        temp_K = np.delete(temp_K, dirichlet_node.index - i, axis=1)

    return temp_K, temp_b


def calculate_fem(K, b, conditions):
    """Calculates the FEM value

    :K: global K
    :b: global b
    :conditions: the list of dirichlet conditions to be applied

    :returns: the result for T

    """
    # get the inverse of K
    K_inv = np.linalg.inv(K)
    T = K_inv.dot(b)

    # insert dirichlet condition values
    for condition in conditions:
        T = np.insert(T, condition.node.index, condition.value)

    return list(map(lambda x: round(x, 2), T))

def post_processing_input(filename, T):
    """Write the GiD post processing input data

    :filename: name of the input file
    :T: list of T for each node

    """
    with open(f"application/data/{filename}.post.res", "w") as f_out:
        f_out.write("GiD Post Results File 1.0\n")
        f_out.write('Result "Temperature" "Load Case 1" 1 Scalar OnNodes\n')
        f_out.write('ComponentNames "T"\n')
        f_out.write("Values\n")

        for i, value in enumerate(T):
            f_out.write("{0}\t{1}\n".format(i + 1, value))
        f_out.write("End values")