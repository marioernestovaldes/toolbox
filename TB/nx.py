import plotly.graph_objects as go
import networkx as nx


def plotly_networkx_graph(G, pos=None, title=None, node_colors=None):
    """
    Create a Plotly visualization of a NetworkX graph.

    Args:
        G: networkx.Graph - The NetworkX graph to visualize.
        pos: dict, optional - A dictionary specifying node positions (default is None).
        title: str, optional - The title of the graph (default is None).
        node_colors: str, optional - The attribute of nodes used for coloring (default is None).

    Returns:
        None: Displays the Plotly figure.
    """
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    # Extract edge coordinates for plotting.
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Extract node coordinates and colors for plotting.
    node_x = []
    node_y = []
    node_c = list(nx.get_node_attributes(G, node_colors).values())

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=list(G.nodes),
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_c,
            size=10,
            colorbar=dict(thickness=15, title="", xanchor="left", titleside="right"),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig.show()
