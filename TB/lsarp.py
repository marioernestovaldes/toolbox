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
    elif organism in ['GAS', 'GBS', 'GCS', 'GDS', 'GGS']:
        # For organisms with 3-letter codes, the plate identifier is the first 6 characters of the sample identifier.
        return x[:6]
    elif organism in ['FAES', 'FAEM']:
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
        'EC': {
            'ORG_SHORT_NAME': 'EC',
            'ORG_LONG_NAME': 'Escherichia coli',
            'QC_A_STRAIN': 'ATCC_25922',
            'QC_D_STRAIN': 'ATCC_BAA196',
            'pipeline_slug': 'escherichia-coli',
            'ORG_COLOR': '#7391f5'
        },
        'FAES': {
            'ORG_SHORT_NAME': 'FAES',
            'ORG_LONG_NAME': 'Enterococcus faecalis',
            'QC_A_STRAIN': 'ATCC_29212',
            'QC_D_STRAIN': 'ATCC_51299',
            'pipeline_slug': 'enterococcus-faecalis',
            'ORG_COLOR': '#fc7e97'
        },
        'FAEM': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'pipeline_slug': 'enterococcus-faecium',
            'ORG_COLOR': '#0cb2f0'
        },
        'GAS': {
            'ORG_SHORT_NAME': 'GAS',
            'ORG_LONG_NAME': 'Group A Streptococcus',
            'QC_A_STRAIN': 'ATCC_19615',
            'QC_D_STRAIN': 'ATCC_12344',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#EE8866'
        },
        'KO': {
            'ORG_SHORT_NAME': 'KO',
            'ORG_LONG_NAME': 'Klebsiella oxytoca',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#c24a4a'
        },
        'KP': {
            'ORG_SHORT_NAME': 'KP',
            'ORG_LONG_NAME': 'Klebsiella pneumoniae',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'pipeline_slug': 'klebsiella-pneumoniae',
            'ORG_COLOR': '#44BB99'
        },
        'PA': {
            'ORG_SHORT_NAME': 'PA',
            'ORG_LONG_NAME': 'Pseudomonas aeruginosa',
            'QC_A_STRAIN': '',
            'QC_D_STRAIN': '',
            'pipeline_slug': 'pseudomonas-aeruginosa',
            'ORG_COLOR': '#fc7e97'
        },
        'SA': {
            'ORG_SHORT_NAME': 'SA',
            'ORG_LONG_NAME': 'Staphylococcus aureus',
            'QC_A_STRAIN': 'ATCC_25923',
            'QC_D_STRAIN': 'ATCC_43300',
            'pipeline_slug': 'staphylococcus-aureus',
            'ORG_COLOR': '#EE8866'
        }
    }

    # Retrieve and return information about the specified organism
    return ORGS[organism]

# def get_treatment():
#     from collections import defaultdict
#     from datetime import datetime
#
#     def calculate_dosing_summary(antibiotics, timestamps):
#         dosing_data = defaultdict(list)
#         summary = []
#
#         for antibiotic, timestamp in zip(antibiotics, timestamps):
#             dosing_data[antibiotic].append(datetime.strptime(timestamp, "%d%b%Y:%H:%M:%S"))
#
#         for antibiotic, doses in dosing_data.items():
#             if len(doses) < 2:
#                 summary.append(f"{antibiotic} irregular dosing")
#             else:
#                 time_gaps = [(doses[i] - doses[i - 1]).total_seconds() / 3600 for i in range(1, len(doses))]
#                 avg_time_gap = sum(time_gaps) / len(time_gaps)
#                 summary.append(f"{antibiotic} every ~{int(avg_time_gap)} hours")
#
#         return ", ".join(summary)
#
#     # Example data
#     antibiotics = df1['drug_name'].to_list()
#     timestamps = df1['ADMINISTERED_DT']
#
#     dosing_summary = calculate_dosing_summary(antibiotics, timestamps)
#     print(dosing_summary)
