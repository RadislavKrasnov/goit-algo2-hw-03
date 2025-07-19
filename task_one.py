import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

edges = [
    ['T1', 'S1', 25],
    ['T1', 'S2', 20],
    ['T1', 'S3', 15],
    ['T2', 'S3', 15],
    ['T2', 'S4', 30],
    ['T2', 'S2', 10],
    ['S1', 'M1', 15],
    ['S1', 'M2', 10],
    ['S1', 'M3', 20],
    ['S2', 'M4', 15],
    ['S2', 'M5', 10],
    ['S2', 'M6', 25],
    ['S3', 'M7', 20],
    ['S3', 'M8', 15],
    ['S3', 'M9', 10],
    ['S4', 'M10', 20],
    ['S4', 'M11', 10],
    ['S4', 'M12', 15],
    ['S4', 'M13', 5],
    ['S4', 'M14', 10],
]



def build_graph(edges):
    G = nx.DiGraph()

    for edge in edges:
        G.add_edge(edge[0], edge[1], capacity=edge[2])

    G.add_node('S')
    G.add_node('T')

    for t in ['T1', 'T2']:
        G.add_edge('S', t, capacity=1000)

    for m in [node for node in G.nodes if node.startswith('M')]:
        G.add_edge(m, 'T', capacity=1000)

    return G

def visualize_graph(graph):
    G_vis = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        if not (u == 'S' or v == 'T' or v == 'T' or (u.startswith('M') and v == 'T')):
            G_vis.add_edge(u, v, capacity=d['capacity'])

    pos = {
        'T1': (-3, 2), 'T2': (-3, -2),
        'S1': (-1.5, 3), 'S2': (-1.5, 1), 'S3': (-1.5, -1), 'S4': (-1.5, -3),
        'M1': (1, 4), 'M2': (1, 3), 'M3': (1, 2),
        'M4': (1, 1.5), 'M5': (1, 1), 'M6': (1, 0.5),
        'M7': (1, -0.5), 'M8': (1, -1), 'M9': (1, -1.5),
        'M10': (1, -2), 'M11': (1, -2.5), 'M12': (1, -3), 'M13': (1, -3.5), 'M14': (1, -4)
    }

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G_vis, pos, node_color='skyblue', node_size=1000)
    nx.draw_networkx_edges(G_vis, pos, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G_vis, pos, font_size=10, font_weight='bold')

    edge_labels = {(u, v): f"{d['capacity']}" for u, v, d in G_vis.edges(data=True)}
    nx.draw_networkx_edge_labels(G_vis, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Логістична мережа: Термінали → Складами → Магазинами", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def bfs(capacity, flow, source, sink, parent):
    visited = [False] * len(capacity)
    queue = deque([source])
    visited[source] = True

    while queue:
       current_node = queue.popleft()

       for neighbor in range(len(capacity)):
           if not visited[neighbor] and capacity[current_node][neighbor] - flow[current_node][neighbor] > 0:
                parent[neighbor] = current_node
                visited[neighbor] = True

                if neighbor == sink:
                    return True
             
                queue.append(neighbor)
    
    return False


def edmonds_karp(capacity, source, sink):
    num_nodes = len(capacity)
    flow = [[0] * num_nodes for _ in range(num_nodes)]
    parent = [-1] * num_nodes
    max_flow = 0

    while bfs(capacity, flow, source, sink, parent):
        path_flow = float('Inf')
        currrent_node = sink

        while currrent_node != source:
            previous_node = parent[currrent_node]
            path_flow = min(path_flow, capacity[previous_node][currrent_node] - flow[previous_node][currrent_node])
            currrent_node = previous_node
        
        currrent_node = sink
        
        while currrent_node != source:
            previous_node = parent[currrent_node]
            flow[previous_node][currrent_node] += path_flow
            flow[currrent_node][previous_node] -= path_flow
            currrent_node = previous_node

        max_flow += path_flow

    return max_flow, flow


def get_node_indices(graph):
    nodes = list(graph.nodes)
    return {node: i for i, node in enumerate(nodes)}


def get_capacity(graph):
    nodes = list(graph.nodes)
    node_indices = get_node_indices(graph)
    n = len(nodes)
    capacity = [[0] * n for _ in range(n)]

    for u, v, d in graph.edges(data=True):
        i, j = node_indices[u], node_indices[v]
        capacity[i][j] = d['capacity']
    
    return capacity


def get_node_index_in_graph(graph, node):
    node_indices = get_node_indices(graph)
    return node_indices[node]


def get_report(graph, flow_matrix):
    nodes = list(graph.nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    terminals = ['T1', 'T2']
    warehouses = ['S1', 'S2', 'S3', 'S4']
    rows = []
    for w in warehouses:
        w_i = idx[w]
        in_flows = {t: flow_matrix[idx[t]][w_i] for t in terminals}
        total_in = sum(in_flows.values())
        if total_in <= 0:
            continue
        for m in nodes:
            if not m.startswith('M'):
                continue
            f_ws = flow_matrix[w_i][idx[m]]
            if f_ws <= 0:
                continue
            for t, f_tw in in_flows.items():
                if f_tw <= 0:
                    continue
                allocated = round(f_ws * (f_tw / total_in))
                if allocated > 0:
                    rows.append({
                        'Термінал': t,
                        'Магазин': m,
                        'Фактичний Потік (одиниць)': allocated
                    })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(['Термінал', 'Магазин']).reset_index(drop=True)
    flow_summary = (
        df_sorted
        .groupby('Термінал', as_index=False)['Фактичний Потік (одиниць)']
        .sum()
    )
    return df_sorted, flow_summary


G = build_graph(edges)

visualize_graph(G)

capacity = get_capacity(G)
source_node = 'S'
sink_node = 'T'
source = get_node_index_in_graph(G, source_node)
sink = get_node_index_in_graph(G, sink_node)
flow_value, flow = edmonds_karp(capacity, source, sink)

print("Максимальний потік:", flow_value)

df_sorted, flow_summary = get_report(G, flow)

print(df_sorted)
print()
print(flow_summary)
