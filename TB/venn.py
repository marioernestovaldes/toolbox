import numpy as np
from matplotlib_venn import venn2_circles, venn2, venn3_circles, venn3

def venn_diagram(a, b, c=None, labels=None, colors=None, **kwargs):
    """
    Create a Venn diagram to visualize the intersections of up to three sets.

    Parameters:
    - a, b, c: Sets or lists to be compared (up to three).
    - labels: Labels for the sets (list of up to three labels).
    - colors: Colors for the set regions (list of up to three colors).
    - **kwargs: Additional keyword arguments for the venn diagram.

    Returns:
    - A Matplotlib Venn diagram object.
    """
    if colors is None:
        colors = ['r', 'y', 'b']
    if c is None:
        venn = _venn_diagram2(a, b, set_labels=labels, set_colors=colors, **kwargs)
    else:
        venn = _venn_diagram3(a, b, c, set_labels=labels, set_colors=colors, **kwargs)
    return venn

def _venn_diagram3(a, b, c, **kwargs):
    """
    Create a Venn diagram for three sets.

    Parameters:
    - a, b, c: Lists or sets for comparison.
    - **kwargs: Additional keyword arguments for the venn3 function.

    Returns:
    - A Matplotlib Venn diagram object for three sets.
    """
    a = list(set(a))
    b = list(set(b))
    c = list(set(c))

    # Calculate the intersections and unique elements for each set.
    only_a = len([x for x in a if x not in b + c])
    only_b = len([x for x in b if x not in a + c])
    only_c = len([x for x in c if x not in a + b])
    a_b = len(np.intersect1d(a, b))
    a_c = len(np.intersect1d(a, c))
    b_c = len(np.intersect1d(b, c))
    a_b_c = len([x for x in a if (x in b) and (x in c)])

    # Create the Venn diagram for three sets.
    venn3(
        subsets=(only_a, only_b, a_b - a_b_c, only_c, a_c - a_b_c, b_c - a_b_c, a_b_c),
        **kwargs
    )
    venn3_circles(
        subsets=(only_a, only_b, a_b - a_b_c, only_c, a_c - a_b_c, b_c - a_b_c, a_b_c),
        linestyle="dashed",
        linewidth=1,
    )

def _venn_diagram2(a, b, **kwargs):
    """
    Create a Venn diagram for two sets.

    Parameters:
    - a, b: Lists or sets for comparison.
    - **kwargs: Additional keyword arguments for the venn2 function.

    Returns:
    - A Matplotlib Venn diagram object for two sets.
    """
    a = list(set(a))
    b = list(set(b))

    # Calculate the intersections and unique elements for each set.
    only_a = len([x for x in a if x not in b])
    only_b = len([x for x in b if x not in a])
    a_b = len(np.intersect1d(a, b))

    # Create the Venn diagram for two sets.
    venn2(subsets=(only_a, only_b, a_b), **kwargs)
    venn2_circles(subsets=(only_a, only_b, a_b), linestyle="dashed", linewidth=1)
