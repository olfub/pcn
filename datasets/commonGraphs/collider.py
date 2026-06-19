import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_collider_scm(alpha=3.0, beta=2.0, gamma=1.0, _visualize=True):
    G = nx.DiGraph()

    # Edges: X -> Y (direct), X -> Z and Y -> Z (collider)
    edges = [
        ('X', 'Y', {'weight': beta}),
        ('X', 'Z', {'weight': alpha}),
        ('Y', 'Z', {'weight': gamma})
    ]
    G.add_edges_from(edges)

    if _visualize:
        pos = {'X': (-1, 1), 'Y': (1, 1), 'Z': (0, 0)}
        plt.figure(figsize=(7, 5))
        nx.draw(G, pos, with_labels=True, node_size=2500, node_color='salmon',
                font_size=12, font_weight='bold', arrowsize=20)

        labels = {(u, v): f"w={d['weight']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black')

        plt.title("Collider SCM\n(Adjusting for Z creates spurious correlation)")
        plt.show()
    return G


def verify_collider_numpy(alpha=3.0, beta=2.0, gamma=1.0, samples=1000000):
    # Data Generation using standard Gaussian noise
    ex = np.random.normal(0, 1, samples)
    ey = np.random.normal(0, 1, samples)
    ez = np.random.normal(0, 1, samples)

    x = ex
    y = beta * x + ey
    z = alpha * x + gamma * y + ez

    # --- 1. Correct Regression (Y ~ X) ---
    # Reshape for NumPy (samples, features)
    X_mat = np.vstack([x, np.ones(samples)]).T
    # Solve for weights using least squares
    w_correct, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    y_hat_correct = X_mat @ w_correct
    mse_correct = np.mean((y - y_hat_correct) ** 2)

    # --- 2. Erroneous Regression (Y ~ X + Z) ---
    XZ_mat = np.vstack([x, z, np.ones(samples)]).T
    w_erroneous, _, _, _ = np.linalg.lstsq(XZ_mat, y, rcond=None)
    y_hat_erroneous = XZ_mat @ w_erroneous
    mse_erroneous = np.mean((y - y_hat_erroneous) ** 2)

    print(f"--- Verification Results ---")
    print(f"MSE Correct (X only):      {mse_correct:.4f}")
    print(f"MSE Erroneous (X and Z):   {mse_erroneous:.4f}")
    print(f"Error Ratio:               {mse_erroneous / mse_correct:.4f}")


if __name__ == "__main__":
    create_collider_scm()
    verify_collider_numpy()
