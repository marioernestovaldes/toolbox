import plotly.express as px
import plotly.graph_objects as go

def px_heatmap(df, colorscale='jet_r', layout_kws=None):
    """
    Creates a heatmap using Plotly Express with the given DataFrame.

    Parameters:
    - df: The DataFrame containing the data to display.
    - colorscale: The color scale to use for the heatmap (default: 'jet_r').
    - layout_kws: Additional keyword arguments for the plot layout (default: None).

    Returns:
    - A Plotly Figure object representing the heatmap.
    """
    fig = go.Figure(data=go.Heatmap(
            z=df.values,     # Data values for the heatmap.
            y=df.index,      # Y-axis labels, typically row labels.
            x=df.columns,    # X-axis labels, typically column labels.
            colorscale=colorscale  # Color scale for the heatmap.
            )
    )
    fig.update_layout(**layout_kws)  # Apply additional layout settings if provided.
    return fig
