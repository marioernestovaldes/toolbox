import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import distinctipy


def lighten_color(color, lightness_factor):
    """
    Lighten a given color by a specified factor.

    Args:
        color (str): The color to be lightened in the format '#RRGGBB'.
        lightness_factor (float): A value between 0 and 1.0 to specify how much to lighten the color.

    Returns:
        str: The lightened color in the format '#RRGGBB'.
    """
    # Check if the input color is in the expected format: '#RRGGBB'
    if color.startswith('#') and len(color) == 7:
        # Extract the RGB components as integers
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Increase the intensity of each RGB component based on the lightness factor
        r = min(255, r + (255 - r) * lightness_factor)
        g = min(255, g + (255 - g) * lightness_factor)
        b = min(255, b + (255 - b) * lightness_factor)

        # Convert the new RGB values back to hex format
        lightened_color = "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
        return lightened_color
    else:
        # Return the original color if it's not in the expected format
        print(f'WARNING!: {color} not in the expected format. Returning original color.')
        return color


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def get_color_palette(data: list, data_type='qualitative', hexa=True):
    """
    Get a color palette based on the data type.

    Parameters:
    - data (list): The data for which the color palette is generated.
    - data_type (str): The type of color palette to generate ('qualitative', 'sequential', or 'divergent').
    - hexa (bool, optional): If True, return color palette in hexadecimal format. Defaults to True.

    Returns:
    - sns.color_palette or ListedColormap: A color palette suitable for the given data type.

    Usage:
    - Qualitative palettes are best for distinguishing categorical data.
    - Sequential palettes are suitable for ordered data.
    - Divergent palettes are appropriate for data with two contrasting extremes.

    Examples:
    - get_color_palette(['A', 'B', 'C']) returns a qualitative palette for categorical data.
    - get_color_palette(range(5), 'sequential') returns a sequential palette for ordered data.
    - get_color_palette([-2, -1, 0, 1, 2], 'divergent') returns a divergent palette.

    Notes:
    - The function uses seaborn and distinctipy libraries for generating color palettes.
    - For 'qualitative' data, it provides visually distinct colors using distinctipy.
    - For 'sequential' data, it returns a seaborn "Blues" color palette.
    - For 'divergent' data, it returns a predefined divergent color palette.

    """
    if data_type == 'qualitative':
        if len(set(data)) <= 14:
            # Return a second qualitative color palette with up to 10 colors.
            colors = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
                      '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#A59AE6',
                      '#882255', '#44AA99', '#999933', '#AA4499']
            if hexa:
                return sns.color_palette(colors).as_hex()[:len(set(data)) + 1]
            else:
                return sns.color_palette(colors)[:len(set(data)) + 1]
        else:
            # Generate N visually distinct colors using the distinctipy library.
            colors = distinctipy.get_colors(len(set(data)), pastel_factor=1)
            if hexa:
                return sns.color_palette(colors).as_hex()
            else:
                return sns.color_palette(colors)

    elif data_type == 'sequential':
        # Return a sequential color palette from the seaborn library.
        return sns.color_palette("Blues", as_cmap=True)

    elif data_type == 'divergent':
        # Return a predefined divergent color palette.
        colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                  '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        return ListedColormap(colors)


def set_sns_style(seaborn_style='nature'):
    """
    Set a customized Seaborn style with a scientific format.

    Parameters:
    - seaborn_style (str): The style to be set. Options: 'nature', 'science', 'vega-lite'. Default is 'nature'.
    """
    # Set the default Seaborn style
    # sns.set()

    sns.reset_defaults()

    if seaborn_style == 'nature':
        nature = {

            # === FONT SETTINGS ===
            'font.family': 'sans-serif',  # Use sans-serif fonts throughout
            'font.sans-serif': 'Arial',  # Preferred sans-serif font
            'mathtext.fontset': 'dejavusans',  # Match math font with sans-serif
            'font.size': 8,  # Base font size
            'axes.labelsize': 10,  # Axis label font size
            'axes.labelpad': 7.5,  # Padding between axis and label
            'xtick.labelsize': 8,  # X-axis tick label font size
            'ytick.labelsize': 8,  # Y-axis tick label font size
            'legend.fontsize': 8,  # Legend font size

            # === FIGURE OUTPUT SETTINGS ===
            'figure.figsize': (4, 4),  # Default figure size (in inches)
            'figure.dpi': 600.0,  # High-resolution output
            'savefig.bbox': 'tight',  # Trim whitespace when saving
            'savefig.pad_inches': 0.05,  # Minimize padding around figure

            # === AXES AND GRID STYLE ===
            'axes.linewidth': 0.5,  # Thin axes borders
            'grid.linewidth': 0.5,  # Grid line thickness

            # === LINE AND MARKER STYLE ===
            'lines.linewidth': 1.0,  # Default line width
            'lines.markersize': 4,  # Marker size
            'lines.markeredgewidth': 1.0,  # Marker border thickness

            # === TICK STYLE ===
            'xtick.direction': 'out',  # Outward ticks on x-axis
            'ytick.direction': 'out',  # Outward ticks on y-axis
            'xtick.major.size': 3,  # Length of major x-ticks
            'ytick.major.size': 3,  # Length of major y-ticks
            'xtick.major.width': 0.5,  # Width of major x-ticks
            'ytick.major.width': 0.5,  # Width of major y-ticks
            'xtick.labelcolor': 'black',  # X-tick label color
            'ytick.labelcolor': 'black',  # Y-tick label color

            # === BOXPLOT STYLE ===
            'boxplot.boxprops.color': 'black',
            'boxplot.boxprops.linestyle': '-',
            'boxplot.boxprops.linewidth': 0.5,

            'boxplot.capprops.color': 'black',
            'boxplot.capprops.linestyle': '-',
            'boxplot.capprops.linewidth': 0.5,

            'boxplot.whiskerprops.color': 'black',
            'boxplot.whiskerprops.linestyle': '-',
            'boxplot.whiskerprops.linewidth': 0.5,
            'boxplot.whiskers': 1.5,  # Whisker length (IQR multiplier)

            'boxplot.medianprops.color': 'black',
            'boxplot.medianprops.linestyle': '-',
            'boxplot.medianprops.linewidth': 0.5,

            'boxplot.flierprops.color': 'black',
            'boxplot.flierprops.linestyle': 'none',
            'boxplot.flierprops.linewidth': 0.5,
            'boxplot.flierprops.marker': 'd',
            'boxplot.flierprops.markeredgecolor': 'black',
            'boxplot.flierprops.markeredgewidth': 0.5,
            'boxplot.flierprops.markerfacecolor': 'black',
            'boxplot.flierprops.markersize': 0.5,

            'boxplot.meanline': False,
            'boxplot.notch': False,
            'boxplot.patchartist': False,
            'boxplot.showbox': True,
            'boxplot.showcaps': True,
            'boxplot.showfliers': True,
            'boxplot.showmeans': False,
            'boxplot.vertical': True,

            'xtick.minor.visible': False,
            'ytick.minor.visible': False,
        }

        # Set the Seaborn theme to 'ticks' with the custom style parameters
        sns.set_theme(context='paper', style='ticks', rc=nature)

    elif seaborn_style == 'science':
        science = {
            # Set line widths
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.,

            # Remove legend frame
            'legend.frameon': False,

            # Always save figures with a tight layout
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'figure.figsize': (4, 4),

            # Use serif fonts for text and math symbols
            'font.family': 'sans-serif',  # change to 'serif' if we want Times New Roman font
            'font.sans-serif': 'Arial',  # change to 'Times New Roman' if we want Times New Roman font
            'mathtext.fontset': 'dejavuserif',

            # Customize the y-axis ticks
            'ytick.direction': 'out',
            'ytick.major.size': 3,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 1.5,
            'ytick.minor.width': 0.5,
            'ytick.minor.visible': True,

            # Customize the x-axis ticks
            'xtick.direction': 'out',
            'xtick.major.size': 3,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 1.5,
            'xtick.minor.width': 0.5,
            'xtick.minor.visible': True,

        }

        # Set the Seaborn theme to 'ticks' with the custom style parameters
        sns.set_theme(context='paper', style='ticks', rc=science)

    elif seaborn_style == 'vega_lite':
        vegalite = {
            # Define font
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],

            # Customize axes
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.edgecolor': '#222222',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.linewidth': 0.8,

            # Customize legend
            'legend.fontsize': 10,
            'legend.frameon': False,

            # Customize lines
            'lines.linewidth': 2,

            # Customize grid
            'grid.alpha': 0.3,
            'grid.linestyle': '',
            'grid.linewidth': 0.5,

            # Customize x-axis
            'xtick.labelsize': 10,
            'xtick.major.size': 5,
            'xtick.major.width': 1.2,
            'xtick.bottom': True,

            # Customize y-axis
            'ytick.labelsize': 10,
            'ytick.major.size': 5,
            'ytick.minor.size': 3.5,
            'ytick.major.width': 1.2,
            'ytick.left': True,

            # Always save figures with a tight layout
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'figure.figsize': (4, 4),
        }

        # Set the Seaborn theme to 'white' with the custom style parameters
        sns.set_theme(context='paper', style='white', rc=vegalite)
