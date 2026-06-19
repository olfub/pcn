import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_confounded_scm(alpha=2.0, beta=1.0, gamma=2.236, _visualize=False):
    """
    Creates an SCM where simply regressing X on Y while adjusting for
    Z gives an expected (MSE) error of 1, while regressing without Z gives
    and expected error of 2.
    Ratios can be veried via the below `verify_error_ratio` function.
    :param alpha: Z -> X weight
    :param beta: X -> Y weight
    :param gamma: Z -> Y weight
    :param _visualize: draws the created graph
    :return:
    """
    G = nx.DiGraph()

    # Z -> X (alpha), X -> Y (beta), Z -> Y (gamma)
    edges = [
        ('Z', 'X', {'weight': alpha}),
        ('Z', 'Y', {'weight': gamma}),
        ('X', 'Y', {'weight': beta})
    ]
    G.add_edges_from(edges)

    if _visualize:
        # Define layout for a causal look (Z at top, X/Y below)
        pos = {
            'Z': (0, 1),
            'X': (-1, 0),
            'Y': (1, 0)
        }

        # Draw nodes and labels
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
                font_size=15, font_weight='bold', arrowsize=20)

        # Draw edge labels (weights)
        edge_labels = {(u, v): f"w={d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

        plt.title("Confounded Linear SCM", fontsize=15)
        plt.margins(0.2)
        plt.show()

    return G


def verify_error_ratio(alpha=2.0, beta=1.0, gamma=2.236, samples=100000):
    # Generate data
    z = np.random.normal(0, 1, samples)
    x = alpha * z + np.random.normal(0, 1, samples)
    y = beta * x + gamma * z + np.random.normal(0, 1, samples)

    # 1. Correct Adjustment (X and Z)
    # Since we know the true weights, the prediction is exactly beta*X + gamma*Z
    y_hat_adj = beta * x + gamma * z
    mse_adj = np.mean((y - y_hat_adj) ** 2)

    # 2. Naive Regression (X only)
    # We find the best linear fit for X -> Y
    b_naive = np.cov(x, y)[0, 1] / np.var(x)
    y_hat_naive = b_naive * x
    mse_naive = np.mean((y - y_hat_naive) ** 2)

    print(f"MSE Adjusted (Correct): {mse_adj:.4f}")
    print(f"MSE Naive (X only):     {mse_naive:.4f}")
    print(f"Ratio (Naive / Adj):    {mse_naive / mse_adj:.4f}")


if __name__ == "__main__":
    create_confounded_scm(_visualize=True)
    verify_error_ratio()
