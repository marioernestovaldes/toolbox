import os, glob

def get_plate(x, organism):
    """
    Get the plate identifier from a sample identifier based on the organism.

    Args:
        x (str): Sample identifier.
        organism (str): Organism identifier.

    Returns:
        str: The plate identifier.
    """
    # Check the organism to determine the plate identifier length
    if organism in ['SA', 'KP', 'KO', 'EC', 'PA']:
        # For organisms with 2-letter codes, the plate identifier is the first 5 characters of the sample identifier.
        return x[:5]
    elif organism in ['GAS', 'GBS', 'GCS', 'GDS', 'GGS']:
        # For organisms with 3-letter codes, the plate identifier is the first 6 characters of the sample identifier.
        return x[:6]
    elif organism in ['FAES', 'FAEM']:
        # For organisms with 4-letter codes, the plate identifier is the first 7 characters of the sample identifier.
        return x[:7]
    else:
        # If the organism is not recognized, return None or raise an exception, depending on your requirements.
        return None  # Modify this part based on your specific use case


def get_org_info(organism):
    """
    Retrieve information about specific organisms.

    This function returns information about various microbial organisms such as their
    short name, long name, and a color code. It simplifies the process of obtaining
    essential information for data labeling, color-coding, and visualization.

    Args:
        organism (str): A short identifier for the organism of interest.

    Returns:
        dict: A dictionary containing information about the organism, including:
            - 'ORG_SHORT_NAME': Short abbreviation or identifier for the organism.
            - 'ORG_LONG_NAME': Full descriptive name of the organism.
            - 'ORG_COLOR': Hexadecimal color code representing the organism's color.
    """

    # Dictionary containing information about different organisms
    ORGS = {
        'EC': {
            'ORG_SHORT_NAME': 'EC',
            'ORG_LONG_NAME': 'Escherichia coli',
            'QC_A_STRAIN': 'ATCC_25922',
            'QC_D_STRAIN': 'ATCC_BAA196',
            'ORG_COLOR': '#7391f5'
        },
        'FAES': {
            'ORG_SHORT_NAME': 'FAES',
            'ORG_LONG_NAME': 'Enterococcus faecalis',
            'QC_A_STRAIN': 'ATCC_29212',
            'QC_D_STRAIN': 'ATCC_51299',
            'ORG_COLOR': '#fc7e97'
        },
        'FAEM': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'ORG_COLOR': '#0cb2f0'
        },
        'GAS': {
            'ORG_SHORT_NAME': 'GAS',
            'ORG_LONG_NAME': 'Group A Streptococcus',
            'QC_A_STRAIN': '',
            'QC_D_STRAIN': '',
            'ORG_COLOR': '#EE8866'
        },
        'KO': {
            'ORG_SHORT_NAME': 'KO',
            'ORG_LONG_NAME': 'Klebsiella oxytoca',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'ORG_COLOR': '#c24a4a'
        },
        'KP': {
            'ORG_SHORT_NAME': 'KP',
            'ORG_LONG_NAME': 'Klebsiella pneumoniae',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'ORG_COLOR': '#44BB99'
        },
        'PA': {
            'ORG_SHORT_NAME': 'PA',
            'ORG_LONG_NAME': 'Pseudomonas aeruginosa',
            'QC_A_STRAIN': '',
            'QC_D_STRAIN': '',
            'ORG_COLOR': '#fc7e97'
        },
        'SA': {
            'ORG_SHORT_NAME': 'SA',
            'ORG_LONG_NAME': 'Staphylococcus aureus',
            'QC_A_STRAIN': 'ATCC_25923',
            'QC_D_STRAIN': 'ATCC_43300',
            'ORG_COLOR': '#EE8866'
        }
    }

    # Retrieve and return information about the specified organism
    return ORGS[organism]
