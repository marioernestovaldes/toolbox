from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
from Bio import AlignIO
from Bio import SeqIO
import tempfile
from uuid import uuid1 as uuid
import os
import pandas as pd
import numpy as np

def multiple_sequence_alignment(
    records,
    output_fn="/var/www/html/dl/alignment.fasta",
    format="clustal",
    id_prefix="",
    index=None,
):
    """
    Perform multiple sequence alignment using MUSCLE and save the alignment in the specified format.

    Args:
    - records: List of sequences to be aligned. Can be either Bio.SeqRecord objects or plain sequences as strings.
    - output_fn: Path to save the alignment file. Defaults to '/var/www/html/dl/alignment.fasta'.
    - format: Format for saving the alignment (e.g., 'clustal'). Defaults to 'clustal'.
    - id_prefix: Prefix for sequence IDs in the output alignment.
    - index: List of IDs if the input is plain sequences. If None, IDs will be generated automatically.

    Returns:
    - Pandas DataFrame with aligned sequences.

    The function performs multiple sequence alignment using MUSCLE and returns the result as a Pandas DataFrame.
    You can access the alignment file using the provided link or download it from the URL.
    """
    if isinstance(records[0], str):
        if index is None:
            records = [
                SeqRecord(Seq(r), id=f"{id_prefix}-{i:03.0f}")
                for i, r in enumerate(records)
            ]
        else:
            records = [
                SeqRecord(Seq(r), id=f"{_id}")
                for i, (r, _id) in enumerate(zip(records, index))
            ]

    path = tempfile.gettempdir()
    job_id = "msa-" + str(uuid())
    tmp_inputs_fn = os.path.join(path, job_id + ".faa")
    if output_fn == None:
        output_fn = os.path.join(path, job_id + ".fasta")
    tmp_log = os.path.join(path, job_id + ".log")
    SeqIO.write(records, tmp_inputs_fn, "fasta")

    msa = MuscleCommandline(
        input=tmp_inputs_fn, out=output_fn, diags=True, maxiters=1, log=tmp_log
    )
    msa()

    with open(output_fn, "r") as file:
        align = AlignIO.read(file, "fasta")

    lines = align.format("stockholm").split("\n")

    result = []
    index = []
    for line in lines:
        if line.startswith("//"):
            continue
        if line == "":
            continue
        if not line.startswith("#"):
            result.append(list(line.split(" ")[1]))
            index.append(line.split(" ")[0])

    return pd.DataFrame(np.array(result), index=index).sort_index()

def fasta_to_df(fns):
    """
    Convert a list of FASTA files to a Pandas DataFrame.

    Args:
    - fns: List of FASTA file paths or a single file path.

    Returns:
    - Pandas DataFrame with sequence information, including labels, descriptions, sequences, and IDs.

    This function reads sequences from FASTA files and creates a Pandas DataFrame with relevant information.
    It's especially useful for handling sequence data from multiple files and organizing it into a DataFrame.
    """
    if isinstance(fns, str):
        fns = [fns]
    sample = []
    sequence = []
    description = []
    records = []
    ids = []
    for fn in fns:
        for record in SeqIO.parse(fn, "fasta"):
            label = os.path.basename(fn).split(".")[0].replace("-", "_")[:10]
            descr = " ".join(record.description.split()[1:])
            seq = str(record.seq)
            if (
                descr == "hypothetical protein"
                or descr.startswith("putative")
                or descr == ""
            ):
                continue
            sample.append(label)
            sequence.append(seq)
            description.append(descr)
            records.append(record)
            seq = str(record.seq)
            ids.append(record.id)
    df = pd.DataFrame(
        {
            "Label": sample,
            "Description": description,
            "Records": records,
            "Sequence": sequence,
            "ID": ids,
        }
    )
    return df
