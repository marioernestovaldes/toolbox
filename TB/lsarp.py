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
    if organism in ['SA', 'KP', 'Kpneu', 'Kvari', 'KO', 'Koxy', 'Kmich', 'Kgri', 'EC', 'PA']:
        # For organisms with 2-letter codes, the plate identifier is the first 5 characters of the sample identifier.
        return x[:5]
    elif organism in ['GAS', 'GBS', 'GCS', 'GDS', 'GGS', 'CNS', 'Group_Strep', 'Saga', 'Sdys', 'Spyo']:
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
    return res[0] if len(res) == 1 else None


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

    colors = ['#77aadd', '#ee8866', '#eedd88', '#ff4d6d', '#33bbee', '#44bb99', '#bbcc33',
              '#ffb3c1', '#ffba08', '#882255', '#dddddd', '#c9184a', '#bbaaee']

    ORGS = {
        'CNS': {
            'ORG_SHORT_NAME': 'CNS',
            'ORG_MID_NAME': 'S. lugdunensis',
            'ORG_LONG_NAME': 'Staphylococcus lugdunensis',
            'QC_A_STRAIN': 'ATCC_12228',
            'QC_D_STRAIN': 'ATCC_15305',
            'pipeline_slug': 'staphylococcus-aureus',
            'ORG_COLOR': '#7391f5',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': '',  # Column of Interest
            'Col_of_I_colors': {'-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'EC': {
            'ORG_SHORT_NAME': 'EC',
            'ORG_MID_NAME': 'E. coli',
            'ORG_LONG_NAME': 'Escherichia coli',
            'QC_A_STRAIN': 'ATCC_25922',
            'QC_D_STRAIN': 'ATCC_BAA196',
            'pipeline_slug': 'escherichia-coli',
            'ORG_COLOR': '#7391f5',
            'AofI': 'ANTIBIOTIC: Ampicillin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: cc',  # Column of Interest
            'Col_of_I_colors': {'ST131 Cplx': colors[7], 'ST95 Cplx': colors[8],
                                'ST73 Cplx': colors[2], 'ST69 Cplx': colors[1],
                                'ST14 Cplx': colors[4], 'ST12 Cplx': colors[5],
                                'ST10 Cplx': colors[6], '-': 'whitesmoke'},  # Colors assigned to Column of Interest
            'MLST': {'adk': 'adk', 'fumC': 'fumC', 'gyrB': 'gyrB', 'icd': 'icd_2',
                     'mdh': 'mdh_1', 'purA': 'purA', 'recA': 'recA'}
        },
        'FAEM': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_MID_NAME': 'E. faecium',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'pipeline_slug': 'enterococcus-faecium',
            'ORG_COLOR': '#7391f5',  # '#0cb2f0',
            'AofI': 'ANTIBIOTIC: Vancomycin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: clades',  # Column of Interest
            'Col_of_I_colors': {'A1': colors[7], 'A2': colors[2],
                                'B': colors[5], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'FAEM_MS3': {
            'ORG_SHORT_NAME': 'FAEM',
            'ORG_MID_NAME': 'E. faecium',
            'ORG_LONG_NAME': 'Enterococcus faecium',
            'QC_A_STRAIN': 'ATCC_35667',
            'QC_D_STRAIN': 'ATCC_700221',
            'pipeline_slug': 'enterococcus-faecium',
            'ORG_COLOR': '#7391f5',  # '#0cb2f0',
            'AofI': 'ANTIBIOTIC: Vancomycin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: clades',  # Column of Interest
            'Col_of_I_colors': {'A1': colors[7], 'A2': colors[2],
                                'B': colors[5], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'FAES': {
            'ORG_SHORT_NAME': 'FAES',
            'ORG_MID_NAME': 'E. faecalis',
            'ORG_LONG_NAME': 'Enterococcus faecalis',
            'QC_A_STRAIN': 'ATCC_29212',
            'QC_D_STRAIN': 'ATCC_51299',
            'pipeline_slug': 'enterococcus-faecalis',
            'ORG_COLOR': '#ee8866',
            'AofI': 'ANTIBIOTIC: Gentamicin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'179': colors[0], '40': colors[1], '16': colors[2],
                                '6': colors[3], '64': colors[4], '103': colors[5],
                                '778': colors[6], '-': 'whitesmoke'},  # Colors assigned to Column of Interest
            MLST: {'aroE': 'aroE', 'gdh': 'zwf', 'gki': 'glcK', 'gyd': 'gap_2',
                   'pstS': 'pstS1_1', 'xpt': 'xpt', 'yqiL': 'fadA'}
        },
        'GAS': {
            'ORG_SHORT_NAME': 'GAS',
            'ORG_MID_NAME': 'S. pyogenes',
            'ORG_LONG_NAME': 'Streptococcus pyogenes',
            'QC_A_STRAIN': 'ATCC_19615',
            'QC_D_STRAIN': 'ATCC_12344',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbaaee',
            'AofI': 'ANTIBIOTIC: Erythromycin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'28': colors[0], '172': colors[1], '120': colors[2],
                                '52': colors[3], '433': colors[4], '36': colors[5],
                                '15': colors[6], '-': 'whitesmoke'},  # Colors assigned to Column of Interest
            MLST: {'gki': 'glcK', 'gtr': 'glnQ_2', 'murI': 'murI', 'mutS': 'mutS',
                   'recP': 'tkt', 'xpt': 'xpt', 'yqiL': 'thlA_2'}
        },
        'GBS': {
            'ORG_SHORT_NAME': 'GBS',
            'ORG_MID_NAME': 'S. agalactiae',
            'ORG_LONG_NAME': 'Streptococcus agalactiae',
            'QC_A_STRAIN': 'ATCC_13813',
            'QC_D_STRAIN': 'ATCC_12386',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#a9dfbf',
            'AofI': 'ANTIBIOTIC: Erythromycin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'1': colors[0], '23': colors[1], '17': colors[2],
                                '8': colors[3], '19': colors[4], '12': colors[5],
                                '459': colors[6], '-': 'whitesmoke'},  # Colors assigned to Column of Interest
            MLST: {'adhP': 'adhA', 'atr': 'group_882', 'glcK': 'glcK', 'glnA': 'glnA',
                   'pheS': 'pheS', 'sdhA': 'sdhA', 'tkt': 'tkt'}
        },
        'GCS': {
            'ORG_SHORT_NAME': 'GCS',
            'ORG_MID_NAME': 'S. dysgalactiae',
            'ORG_LONG_NAME': 'Streptococcus dysgalactiae',
            'QC_A_STRAIN': 'ATCC_12388',
            'QC_D_STRAIN': 'ATCC_35666',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#eedd88',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': '',  # Column of Interest
            'Col_of_I_colors': {'-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'GDS': {
            'ORG_SHORT_NAME': 'GDS',
            'ORG_MID_NAME': 'S. pasteurianus',
            'ORG_LONG_NAME': 'Streptococcus pasteurianus',
            'QC_A_STRAIN': 'ATCC_9809',
            'QC_D_STRAIN': 'ATCC_33317',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#f4a261',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': '',  # Column of Interest
            'Col_of_I_colors': {'-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'GGS': {
            'ORG_SHORT_NAME': 'GGS',
            'ORG_MID_NAME': 'S. dysgalactiae',
            'ORG_LONG_NAME': 'Streptococcus dysgalactiae',
            'QC_A_STRAIN': 'ATCC_12394A',
            'QC_D_STRAIN': 'ATCC_12394D',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbcc33',
            'AofI': 'ANTIBIOTIC: Erythromycin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'17': colors[0], '29': colors[1], '12': colors[2],
                                '15': colors[3], '8': colors[4], '282': colors[5],
                                '63': colors[12], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Group_Strep': {
            'ORG_SHORT_NAME': 'GS',
            'ORG_MID_NAME': 'Group_Strep',
            'ORG_LONG_NAME': 'Group Streptococcus',
            'QC_A_STRAIN': '',
            'QC_D_STRAIN': '',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#a9dfbf',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Stre agalactiae': '#eedd88', 'Stre dysgalactiae': '#9cee98',
                                'Stre pyogenes': '#bbaaee'}  # Colors assigned to Column of Interest
        },
        'Spyo': {
            'ORG_SHORT_NAME': 'Spyo',
            'ORG_MID_NAME': 'S. pyogenes',
            'ORG_LONG_NAME': 'Streptococcus pyogenes',
            'QC_A_STRAIN': 'ATCC_19615',
            'QC_D_STRAIN': 'ATCC_12344',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbaaee',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Stre agalactiae': '#eedd88', 'Stre dysgalactiae': '#9cee98',
                                'Stre pyogenes': '#bbaaee', '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Saga': {
            'ORG_SHORT_NAME': 'Saga',
            'ORG_MID_NAME': 'S. agalactiae',
            'ORG_LONG_NAME': 'Streptococcus agalactiae',
            'QC_A_STRAIN': 'ATCC_13813',
            'QC_D_STRAIN': 'ATCC_12386',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#a9dfbf',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Stre agalactiae': '#eedd88', 'Stre dysgalactiae': '#9cee98',
                                'Stre pyogenes': '#bbaaee', '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Sdys': {
            'ORG_SHORT_NAME': 'Sdys',
            'ORG_MID_NAME': 'S. dysgalactiae',
            'ORG_LONG_NAME': 'Streptococcus dysgalactiae',
            'QC_A_STRAIN': 'ATCC_12394A',
            'QC_D_STRAIN': 'ATCC_12394D',
            'pipeline_slug': 'group-streptococcus',
            'ORG_COLOR': '#bbcc33',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Stre agalactiae': '#eedd88', 'Stre dysgalactiae': '#9cee98',
                                'Stre pyogenes': '#bbaaee', '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'KO': {
            'ORG_SHORT_NAME': 'KOc',
            'ORG_MID_NAME': 'K. oxytoca complex',
            'ORG_LONG_NAME': 'Klebsiella oxytoca complex',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#bbaaee',
            'AofI': 'ANTIBIOTIC: Cefazolin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Klebsiella oxytoca': '#bbaaee',
                                'Klebsiella michiganensis': '#ffb5c2',
                                'Klebsiella grimontii': '#a9dfbf',

                                # https://pmc.ncbi.nlm.nih.gov/articles/PMC8635272/
                                # K. pasteurii represents the phylogroup Ko4. Strain SPARK_836_C1T, a representative
                                # Ko4 strain, had the highest ANI value, 95.5%, with K. grimontii 06D021T, which falls
                                # into the 95% to 96% inconclusive zone of defining a bacterial species

                                # We performed an analysis and found that the isDDH between
                                # K.pasteurii SPARK_836_C1T and K.grimontii 06D021T was 67.8 %, below the
                                # 70 % cutoff(23).The species status of K.pasteurii is therefore confirmed.

                                # 'Klebsiella pasteurii': '#eedd88',
                                '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Koxy': {
            'ORG_SHORT_NAME': 'Koxy',
            'ORG_MID_NAME': 'K. oxytoca',
            'ORG_LONG_NAME': 'Klebsiella oxytoca',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#fc7e97',
            'AofI': 'ANTIBIOTIC: Cefazolin',  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC105771
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'199': colors[12], '176': colors[8], '223': colors[2],
                                '21': colors[1], '37': colors[4], '53': colors[5],
                                '58': colors[6], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Kmich': {
            'ORG_SHORT_NAME': 'Koxy',
            'ORG_MID_NAME': 'K. michiganensis',
            'ORG_LONG_NAME': 'Klebsiella michiganensis',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#ee8866',
            'AofI': 'ANTIBIOTIC: Cefazolin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'27': colors[7], '11': colors[8], '29': colors[2],
                                '44': colors[12], '85': colors[4], '213': colors[5],
                                '50': colors[6], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'Kgri': {
            'ORG_SHORT_NAME': 'Koxy',
            'ORG_MID_NAME': 'K. grimontii',
            'ORG_LONG_NAME': 'Klebsiella grimontii',
            'QC_A_STRAIN': 'ATCC_700324',
            'QC_D_STRAIN': 'ATCC_51983',
            'pipeline_slug': 'klebsiella-oxytoca',
            'ORG_COLOR': '#bbaaee',
            'AofI': 'ANTIBIOTIC: Cefazolin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'216': colors[7], '186': colors[8], '431': colors[2],
                                '184': colors[1], '215': colors[4], '168': colors[5],
                                '214': colors[6], '-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'KP': {
            'ORG_SHORT_NAME': 'KPc',
            'ORG_MID_NAME': 'K. pneumoniae complex',
            'ORG_LONG_NAME': 'Klebsiella pneumoniae complex',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'pipeline_slug': 'klebsiella-pneumoniae',
            'ORG_COLOR': '#fc7e97',
            'AofI': 'ANTIBIOTIC: Trimethoprim-sulfamethoxazole',  # Antibiotic of Interest
            'Col_of_I': 'GENO: species',  # Column of Interest
            'Col_of_I_colors': {'Klebsiella pneumoniae': '#fc7e97', # Colors assigned to Column of Interest
                                'Klebsiella variicola subsp. variicola': '#bbcc33',
                                'Klebsiella quasipneumoniae subsp. similipneumoniae': '#eedd88',
                                'Klebsiella quasipneumoniae subsp. quasipneumoniae': '#77aadd',
                                '-': 'whitesmoke'}
        },
        'Kpneu': {
            'ORG_SHORT_NAME': 'KPc',
            'ORG_MID_NAME': 'K. pneumoniae',
            'ORG_LONG_NAME': 'Klebsiella pneumoniae',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'pipeline_slug': 'klebsiella-pneumoniae',
            'ORG_COLOR': '#a9dfbf',
            'AofI': 'ANTIBIOTIC: Trimethoprim-sulfamethoxazole',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'23': colors[7], '20': colors[8], '37': colors[2],
                                '253': colors[1], '17': colors[4], '45': colors[5],
                                '36': colors[6], '-': 'whitesmoke'}
        },
        'Kvari': {
            'ORG_SHORT_NAME': 'KPc',
            'ORG_MID_NAME': 'K. variicola',
            'ORG_LONG_NAME': 'Klebsiella variicola',
            'QC_A_STRAIN': 'ATCC_700603',
            'QC_D_STRAIN': 'ATCC_BAA1705',
            'pipeline_slug': 'klebsiella-pneumoniae',
            'ORG_COLOR': '#bbcc33',
            'AofI': 'ANTIBIOTIC: Trimethoprim-sulfamethoxazole',  # Antibiotic of Interest
            'Col_of_I': 'GENO: mlst',  # Column of Interest
            'Col_of_I_colors': {'347': colors[7], '2011': colors[8], '1562': colors[2],
                                '208': colors[1], '919': colors[4], '355': colors[5],
                                '641': colors[6], '-': 'whitesmoke'}
        },
        'PA': {
            'ORG_SHORT_NAME': 'PA',
            'ORG_MID_NAME': 'P. aeruginosa',
            'ORG_LONG_NAME': 'Pseudomonas aeruginosa',
            'QC_A_STRAIN': 'ATCC_27853',
            'QC_D_STRAIN': 'BAA_2795',
            'pipeline_slug': 'pseudomonas-aeruginosa',
            'ORG_COLOR': '#0173b2',
            'AofI': '',  # Antibiotic of Interest
            'Col_of_I': '',  # Column of Interest
            'Col_of_I_colors': {'-': 'whitesmoke'}  # Colors assigned to Column of Interest
        },
        'SA': {
            'ORG_SHORT_NAME': 'SA',
            'ORG_MID_NAME': 'S. aureus',
            'ORG_LONG_NAME': 'Staphylococcus aureus',
            'QC_A_STRAIN': 'ATCC_25923',
            'QC_D_STRAIN': 'ATCC_43300',
            'pipeline_slug': 'staphylococcus-aureus',
            'ORG_COLOR': '#fc7e97',
            'AofI': 'ANTIBIOTIC: Cloxacillin',  # Antibiotic of Interest
            'Col_of_I': 'GENO: cc',  # Column of Interest
            'Col_of_I_colors': {'CC30': colors[12], 'CC5': colors[8], 'CC8': colors[2],
                                'CC45': colors[1], 'CC15': colors[4], 'CC97': colors[5],
                                'CC1': colors[6], '-': 'whitesmoke'
                                # 'CC22': colors[7]
                                },
            'MSLT': {'arcC': 'arcC2', 'aroE': 'aroE', 'glpF': 'glpF', 'gmk': 'gmk',
                     'pta': 'pta', 'tpi': 'tpiA', 'yqiL': 'group_4471'}
        }
    }

    # Retrieve and return information about the specified organism
    return ORGS[organism]
