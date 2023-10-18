import seaborn as sns
import matplotlib.pyplot as plt


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
        return color


def set_sns_style(style='ticks'):
    """
    Set a customized Seaborn style with a scientific format.
    """
    # Set the default Seaborn style
    sns.set()

    # Define custom style parameters for a scientific format
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

        # Use serif fonts for text and math symbols
        'font.family': 'serif',
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
        'xtick.minor.visible': True
    }

    # Set the Seaborn theme to 'ticks' with the custom style parameters
    sns.set_theme(style=style, rc=science)
