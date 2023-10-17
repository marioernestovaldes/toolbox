from IPython.core.display import display, HTML

def notebook_width(width):
    # Display an HTML style element to adjust the container's width
    display(HTML("<style>.container { width:%d%% !important; }</style>" % width))
