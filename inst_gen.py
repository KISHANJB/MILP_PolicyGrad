import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

def generate_setcover(
        nrows=500,
        ncols=1000,
        density=0.05,
        max_coef=100,
        seed=0):

    rng = np.random.RandomState(seed)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()

    model = gp.Model(env=env)

    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows
    assert nnzrs >= 2 * ncols

    # ---------------------------------------------------------
    # This follows the instance-generation procedure used by
    # Gasse et al. for their Set Cover instances.
    # ---------------------------------------------------------

    indices = rng.choice(ncols, size=nnzrs)

    # Force at least two rows per column
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)

    _, col_nrows = np.unique(indices, return_counts=True)

    # Force at least one column per row
    indices[:nrows] = rng.permutation(nrows)

    i = 0
    indptr = [0]

    for n in col_nrows:

        if i >= nrows:

            indices[i:i+n] = rng.choice(
                nrows,
                size=n,
                replace=False
            )

        elif i + n > nrows:

            remaining_rows = np.setdiff1d(
                np.arange(nrows),
                indices[i:nrows],
                assume_unique=True
            )

            indices[nrows:i+n] = rng.choice(
                remaining_rows,
                size=i+n-nrows,
                replace=False
            )

        i += n
        indptr.append(i)

    # Objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # Construct sparse constraint matrix
    A = sp.csc_matrix(
        (
            np.ones(len(indices), dtype=int),
            indices,
            indptr
        ),
        shape=(nrows, ncols)
    ).tocsr()

    # ---------------------------------------------------------
    # YOUR GUROBI MODEL
    # ---------------------------------------------------------

    

    # Binary variables
    x = model.addVars(
        ncols,
        vtype=GRB.BINARY,
        name="x"
    )

    # Objective:
    #       minimize c^T x
    model.setObjective(
        gp.quicksum(c[j] * x[j] for j in range(ncols)),
        GRB.MINIMIZE
    )

    # Constraints:
    #       A x >= 1
    for i in range(nrows):

        cols = A.indices[A.indptr[i]:A.indptr[i+1]]

        model.addConstr(
            gp.quicksum(x[j] for j in cols) >= 1,
            name=f"C{i}"
        )

    model.update()

    return model






def generate_indset(
        n_nodes=500,
        affinity=4,
        seed=0):

    rng = np.random.RandomState(seed)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()

    model = gp.Model(env=env)

    # =========================================================
    # Generate Barabási-Albert graph
    # =========================================================

    assert affinity >= 1
    assert affinity < n_nodes

    edges = set()

    degrees = np.zeros(
        n_nodes,
        dtype=int
    )

    neighbors = {
        node: set()
        for node in range(n_nodes)
    }

    for new_node in range(affinity, n_nodes):

        # First node connects to all previous nodes
        if new_node == affinity:

            neighborhood = np.arange(new_node)

        else:

            neighbor_prob = (
                degrees[:new_node]
                / (2 * len(edges))
            )

            neighborhood = rng.choice(
                new_node,
                affinity,
                replace=False,
                p=neighbor_prob
            )

        for node in neighborhood:

            edge = (node, new_node)

            edges.add(edge)

            degrees[node] += 1
            degrees[new_node] += 1

            neighbors[node].add(new_node)
            neighbors[new_node].add(node)

    # =========================================================
    # Greedy clique partition
    #
    # This follows the idea used by the Gasse generator:
    # replace individual edge constraints by clique
    # inequalities when possible.
    # =========================================================

    cliques = []

    leftover_nodes = (
        -degrees
    ).argsort().tolist()

    while leftover_nodes:

        clique_center = leftover_nodes[0]
        leftover_nodes = leftover_nodes[1:]

        clique = {clique_center}

        candidate_neighbors = (
            neighbors[clique_center]
            .intersection(leftover_nodes)
        )

        densest_neighbors = sorted(
            candidate_neighbors,
            key=lambda x: -degrees[x]
        )

        for neighbor in densest_neighbors:

            if all(
                neighbor in neighbors[clique_node]
                for clique_node in clique
            ):
                clique.add(neighbor)

        cliques.append(clique)

        leftover_nodes = [
            node
            for node in leftover_nodes
            if node not in clique
        ]

    # =========================================================
    # Construct inequalities
    # =========================================================

    inequalities = set(edges)

    for clique in cliques:

        clique = tuple(sorted(clique))

        # Remove pairwise edge constraints inside clique
        for edge in combinations(clique, 2):

            if edge in inequalities:
                inequalities.remove(edge)

        # Replace by one clique inequality
        if len(clique) > 1:
            inequalities.add(clique)

    # =========================================================
    # Gurobi model
    # =========================================================

    

    x = model.addVars(
        n_nodes,
        vtype=GRB.BINARY,
        name="x"
    )

    # Maximum independent set:
    #
    #       max sum_i x_i
    #
    model.setObjective(
        -gp.quicksum(x[i] for i in range(n_nodes)),
        GRB.MINIMIZE
    )

    # Clique / edge constraints
    for k, group in enumerate(inequalities):

        model.addConstr(
            gp.quicksum(x[i] for i in group) <= 1,
            name=f"C{k}"
        )

    model.update()

    return model


