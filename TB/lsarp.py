import os, glob
import re

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
    elif organism in ['GAS', 'GBS', 'GCS', 'GDS', 'GGS', 'CNS']:
        # For organisms with 3-letter codes, the plate identifier is the first 6 characters of the sample identifier.
        return x[:6]
    elif organism in ['FAES', 'FAEM', 'FAEM_MS3']:
        # For organisms with 4-letter codes, the plate identifier is the first 7 characters of the sample identifier.
        return x[:7]
    else:
        # If the organism is not recognized, return None or raise an exception, depending on your requirements.
        return None  # Modify this part based on your specific use case

def get_row(x):
    """
    This function searches for specific patterns in the input string 'x' to identify and extract a row identifier.

    Parameters:
    - x (str): The input string from which the row identifier will be extracted.

    Returns:
    - row_identifier (str): The extracted row identifier (e.g., 'A', 'B', etc.). Returns None if no match is found.

    Example:
    If 'x' is 'PA001_G2_230818_B_T1', this function will return 'B'.
    If 'x' is 'PA001_G2_230818_A', this function will return 'A'.
    If no match is found, the function returns None.
    """

    # Attempt to find a row identifier using a regex pattern
    res = re.findall(r'_[ABCDEFGH]_', x)
    if len(res) == 1:
        return res[0][1]

    # If the first pattern did not match, attempt to find a row identifier with a different pattern
    res = re.findall(r'_[ABCDEFGH]$', x)
    if len(res) == 1:
        return res[0][-1]

    # Return None if no match is found
    return None

def get_date(x):
    """
    This function searches for a date pattern within the input string 'x' and extracts it.
    The date pattern is expected to be a string containing exactly six digits (e.g., '230901').

    Parameters:
    - x (str): The input string from which the date pattern will be extracted.

    Returns:
    - date_pattern (str): The extracted date pattern (e.g., '230901'). Returns None if no match is found.

    Example:
    If 'x' is 'PA001_G2_230818_B_T1', this function will return '230818'.
    If no match is found, the function returns None.
    """

    # Attempt to find a date pattern using a regex pattern
    res = re.findall("[0-9]{6}", x)
    if len(res) == 1:
        return res[0]

    # Return None if no date pattern is found
    return None

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
        'CNS': {
            'ORG_SHORT_NAME': 'CNS',
            'ORG_MID_NAME': 'S. lugdunensis',
            'ORG_LONG_NAME': 'Staphylococcus lugdunensis',
            'QC_A_STRAIN': 'ATCC_12228',
            'QC_D_STRAIN': 'ATCC_15305',
            'pipeline_slug': 'staphylococcus-aureus',
            'ORG_COLOR': '#7391f5'
        },
        'EC': {
            'ORG_SHORT_NAME': 'EC',
            'ORG_MID_NAME': 'E. coli',
            'ORG_LONG_NAME': 'Escherichia coli',
            'QC_A_STRAIN': 'ATCC_25922',
            'QC_D_STRAIN': 'ATCC_BAA196',
            'pipeline_slug': 'escherichia-coli',
            'ORG_COLOR': '#7391f5'
        },
        'FAEM': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_MID_NAME': 'E. faecium',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'pipeline_slug': 'enterococcus-faecium',
            'ORG_COLOR': '#0cb2f0'
        },
        'FAEM_MS3': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_MID_NAME': 'E. faecium',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'pipeline_slug': 'enterococcus-faecium',
            'ORG_COLOR': '#0cb2f0'
        },
        'FAES': {
            'ORG_SHORT_NAME': 'FAES',
            'ORG_MID_NAME': 'E. faecalis',
            'ORG_LONG_NAME': 'Enterococcus faecalis',
            'QC_A_STRAIN': 'ATCC_29212',
            'QC_D_STRAIN': 'ATCC_51299',
            'pipeline_slug': 'enterococcus-faecalis',
            'ORG_COLOR': '#fc7e97'
        },
        'GAS': {
            'ORG_SHORT_NAME': 'GAS',
            'ORG_MID_NAME': 'S. pyogenes',
            'ORG_LONG_NAME': 'Streptococcus pyogenes',
            'QC_A_STRAIN': 'ATCC_19615',
            'QC_D_STRAIN': 'ATCC_12344',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbaaee'
        },
        'GBS': {
            'ORG_SHORT_NAME': 'GBS',
            'ORG_MID_NAME': 'S. agalactiae',
            'ORG_LONG_NAME': 'Streptococcus agalactiae',
            'QC_A_STRAIN': 'ATCC_13813',
            'QC_D_STRAIN': 'ATCC_12386',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#64b7cc'
        },
        'GCS': {
            'ORG_SHORT_NAME': 'GCS',
            'ORG_MID_NAME': 'S. dysgalactiae',
            'ORG_LONG_NAME': 'Streptococcus dysgalactiae',
            'QC_A_STRAIN': 'ATCC_12388',
            'QC_D_STRAIN': 'ATCC_35666',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#eedd88'
        },
        'GDS': {
            'ORG_SHORT_NAME': 'GDS',
            'ORG_MID_NAME': 'S. pasteurianus',
            'ORG_LONG_NAME': 'Streptococcus pasteurianus',
            'QC_A_STRAIN': 'ATCC_9809',
            'QC_D_STRAIN': 'ATCC_33317',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#f4a261'
        },
        'GGS': {
            'ORG_SHORT_NAME': 'GGS',
            'ORG_MID_NAME': 'S. dysgalactiae',
            'ORG_LONG_NAME': 'Streptococcus dysgalactiae',
            'QC_A_STRAIN': 'ATCC_12394A',
            'QC_D_STRAIN': 'ATCC_12394D',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbcc33'
        },
        'Group_Strep': {
            'ORG_SHORT_NAME': 'GS',
            'ORG_MID_NAME': 'Group_Strep',
            'ORG_LONG_NAME': 'Group Streptococcus',
            'QC_A_STRAIN': '',
            'QC_D_STRAIN': '',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#a9dfbf'
        },
        'KO': {
            'ORG_SHORT_NAME': 'KOc',
            'ORG_MID_NAME': 'K. oxytoca complex',
            'ORG_LONG_NAME': 'Klebsiella oxytoca complex',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#c24a4a'
        },
        'KP': {
            'ORG_SHORT_NAME': 'KPc',
            'ORG_MID_NAME': 'K. pneumoniae complex',
            'ORG_LONG_NAME': 'Klebsiella pneumoniae complex',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'pipeline_slug': 'klebsiella-pneumoniae',
            'ORG_COLOR': '#44bb99'
        },
        'PA': {
            'ORG_SHORT_NAME': 'PA',
            'ORG_MID_NAME': 'P. aeruginosa',
            'ORG_LONG_NAME': 'Pseudomonas aeruginosa',
            'QC_A_STRAIN': 'ATCC_27853',
            'QC_D_STRAIN': 'BAA_2795',
            'pipeline_slug': 'pseudomonas-aeruginosa',
            'ORG_COLOR': '#0173b2'
        },
        'SA': {
            'ORG_SHORT_NAME': 'SA',
            'ORG_MID_NAME': 'S. aureus',
            'ORG_LONG_NAME': 'Staphylococcus aureus',
            'QC_A_STRAIN': 'ATCC_25923',
            'QC_D_STRAIN': 'ATCC_43300',
            'pipeline_slug': 'staphylococcus-aureus',
            'ORG_COLOR': '#ee8866'
        }
    }

    # Retrieve and return information about the specified organism
    return ORGS[organism]
