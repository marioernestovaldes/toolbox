import requests

def get_smiles_from_inchikey(inchikey):
    # Define the URL to query PubChem's REST API for the Canonical SMILES using the InChIKey
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON"

    # Send an HTTP GET request to the defined URL and parse the JSON response
    response = requests.get(url).json()

    # Extract and return the Canonical SMILES from the response
    canonical_smiles = response["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    return canonical_smiles
