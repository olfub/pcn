import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from spnn.datasets.binary_bn import BinaryBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


cpdx = np.array([
    0.5,  # Z=0
    0.5,  # Z=1
])

cpdy = np.array([
    0.3,  # X=0 Z=0
    0.9,  # X=0 Z=1
    0.1,  # X=1 Z=0
    0.8  # X=1 Z=1
])
#MSE Adjusted (X, Z): 0.1380
#MSE Naive (X only):  0.2438
#Error Ratio:         1.7668


def verify_binary_confounder(samples=1000000):
    base = 0.5

    z = np.random.binomial(1, base, samples)
    x = np.array(np.random.rand(samples) < np.take(cpdx,z))
    y = np.array(np.random.rand(samples) < np.take(cpdy,2*x+z))

    def get_mse(predictors, target):
        # Linear regression using NumPy
        A = np.column_stack([predictors, np.ones(len(target))])
        w, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
        prediction = A @ w
        return np.mean((target - prediction) ** 2)

    mse_adj = get_mse(np.column_stack([x, z]), y)
    mse_naive = get_mse(x, y)

    print(f"MSE Adjusted (X, Z): {mse_adj:.4f}")
    print(f"MSE Naive (X only):  {mse_naive:.4f}")
    print(f"Error Ratio:         {mse_naive / mse_adj:.4f}")


def create_graph(_visualize=False):
    G = nx.DiGraph()
    G.add_edge('X', 'Y')
    G.add_edge('Z', 'X')
    G.add_edge('Z', 'Y')

    if _visualize:
        pos = {'Z': (0, 1), 'X': (-1, 0), 'Y': (1, 0)}
        plt.figure(figsize=(7, 5))

        nx.draw(G, pos, node_color='lightgreen',
                node_size=2500, font_weight='bold', arrowsize=25)

        nx.draw_networkx_edge_labels(G, pos)
        plt.show()
    return G


def create_confounded_binary_bn():
    digraph = create_graph()
    binaryBN = BinaryBayesianNetwork(digraph)
    model = binaryBN.model

    #student = DiscreteBayesianNetwork([('diff', 'grades'), ('aptitude', 'grades')])
    tcpdZ = TabularCPD("Z", 2, np.array([[0.5],[0.5]]))
    tcpdX = TabularCPD("X",2,
                       np.vstack((cpdx.T,1-cpdx)),
                       evidence=["Z"], evidence_card=[2])
    tcpdY = TabularCPD("Y",2,
                       np.vstack((cpdy.T,1-cpdy)),
                       evidence=["X", "Z"], evidence_card=[2,2])

    model.add_cpds(tcpdZ, tcpdX, tcpdY)

    return binaryBN


if __name__ == "__main__":
    create_graph(_visualize=True)
    create_confounded_binary_bn()
    verify_binary_confounder()
