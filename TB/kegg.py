from Bio.KEGG import REST

def get(ID="K18766", db_id=None):
    """
    Fetch data from the KEGG database.

    Args:
    ID: str
        The KEGG identifier for the entry you want to fetch.
    db_id: str, default None
        The optional database ID for specifying the KEGG database to query.
        - 'ko': Orthology database
        - 'ec': Enzyme database
        - 'cpd': Compound database
        - 'rn': Reaction database

    If db_id is None, the request will be directed based on the first letter of the ID:
        - 'C' -> Compound database
        - 'K' -> Orthology database
        - 'R' -> Reaction database
        All other requests will be directed to the Enzyme database.

    Returns:
    data: list
        A list containing the data fetched from the KEGG database.
    """

    # Define database keys for KEGG databases
    db_keys = {"orthology": "ko", "enzyme": "ec", "compound": "cpd", "reaction": "rn"}

    # Determine the database key to use based on db_id or the first letter of the ID
    if db_id is not None:
        db_key = db_keys[db_id]
    elif ID.startswith("C"):
        db_key = db_keys["compound"]
    elif ID.startswith("K"):
        db_key = db_keys["orthology"]
    elif ID.startswith("R"):
        db_key = db_keys["reaction"]
    else:
        db_key = db_keys["enzyme"]

    # Fetch data from the KEGG database using the specified or inferred database key
    data = REST.kegg_get(f"{db_key}:{ID}").read().split("\n")

    return data
