import networkx as nx
import matplotlib.pyplot as plt

def create_diamond_scm(xa=1,ay=2,xb=3,by=6, _visualize=False):
    """
    Creates an SCM with diamond structure
    ```
        X
      /  \
     A   B
      \ /
       Y
    ```
    :param xa: X->A
    :param xb: X->B
    :param ay: A->Y
    :param by: B->Y
    :param _visualize:
    :return:
    """
    G = nx.DiGraph()

    edges = [
        ('X', 'A', {'weight': xa}),
        ('X', 'B', {'weight': xb}),
        ('A', 'Y', {'weight': ay}),
        ('B', 'Y', {'weight': by}),
    ]
    G.add_edges_from(edges)

    if _visualize:
        # Define layout for a causal look (Z at top, X/Y below)
        pos = {
            'X': (0, 1),
            'A': (-1, 0),
            'B': (1, 0),
            'Y': (0, -1)
        }

        # Draw nodes and labels
        plt.figure(figsize=(6, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
                font_size=15, font_weight='bold', arrowsize=20)

        # Draw edge labels (weights)
        edge_labels = {(u, v): f"w={d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

        plt.title("Diamond Linear SCM", fontsize=15)
        plt.margins(0.2)
        plt.show()
    return G

if __name__ == "__main__":
    create_diamond_scm(_visualize=True)
