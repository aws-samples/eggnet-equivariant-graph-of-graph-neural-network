# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Helpers for parsing protein structure files and generating contact maps.
"""

import gzip
import boto3
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import pairwise_distances
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Entity import Entity as PDBEntity
from rdkit import Chem
from tqdm import tqdm
from .xpdb import SloppyStructureBuilder


def idx_align(pdb_seq: str, og_seq: str):
    """Maps indices of original and PDB sequences to mutual alignment
    Args:
        pdb_seq: Sequence of amino acids as extracted from PDB file
        og_seq: Original sequence of amino acids
    Returns:
        Tuple where the first element is a list of integers containing the indices in
        the PDB sequence for which aligned information is available and the second element
        is a list of integers containing the indices in the original sequence
        for which aligned information is available
    """
    alignment = pairwise2.align.globalxx(Seq(og_seq), Seq(pdb_seq))[0]
    og_seq_align = alignment.seqA
    og_idxs = []
    og_offset = 0

    pdb_seq_align = alignment.seqB
    pdb_idxs = []
    pdb_offset = 0
    for i in range(len(pdb_seq_align)):
        valid = True
        pdb_res = pdb_seq_align[i]
        og_res = og_seq_align[i]
        if pdb_res == "-":
            pdb_offset += 1
            valid = False
        if og_res == "-":
            og_offset += 1
            valid = False
        if valid and (pdb_res == og_res):
            pdb_idxs.append(i - pdb_offset)
            og_idxs.append(i - og_offset)

    return pdb_idxs, og_idxs


def calc_adj_matrix(structure, dist_thresh=10.0):
    """Returns an adjacency matrix for a PDB structure
    Args:
        structure : BioPython PDB Structure object representing a protein containing N residues
        dist_thresh : float, optional. Value specifying threshold for classifying contacts between residues

    Returns:
        Tuple where the first element is a np.ndarray adjacency matrix with shape NxN, where N is the number of residues in `structure` and the second element is np.ndarray NxN matrix specifying distances between residues.
    """
    residues = list(structure.get_residues())
    coords = np.asarray([residue["CA"].coord for residue in residues])
    dist_mat = pairwise_distances(coords, metric="euclidean")
    return 1 * (dist_mat < dist_thresh), dist_mat


def gunzip_to_ram(gzip_file_path):
    """
    gunzip a gzip file and decode it to a io.StringIO object.

    Args:
        gzip_file_path: String. Gunzip filepath.

    Returns:
        io.StringIO object.
    """
    content = []
    with gzip.open(gzip_file_path, "rb") as f:
        for line in f:
            content.append(line.decode("utf-8"))

    temp_fp = StringIO("".join(content))
    return temp_fp


def _parse_structure(parser, name, file_path):
    """Parse a .pdb or .cif file into a structure object.
    The file can be gzipped.

    Args:
        parser: a Bio.PDB.PDBParser or Bio.PDB.MMCIFParser instance.
        name: String. name of protein
        file_path: String. Filpath of the pdb or cif file to be read.

    Retruns:
        a Bio.PDB.Structure object representing the protein structure.

    """
    if pd.isnull(file_path):
        return None
    if file_path.endswith(".gz"):
        structure = parser.get_structure(name, gunzip_to_ram(file_path))
    else:  # not gzipped
        structure = parser.get_structure(name, file_path)
    return structure


parse_pdb_structure = _parse_structure  # for backward compatiblity


def parse_structure(pdb_parser, cif_parser, name, file_path):
    """Parse a .pdb file or .cif file into a structure object.
    The file can be gzipped.

    Args:
        pdb_parser: a Bio.PDB.PDBParser instance
        cif_parser: Bio.PDB.MMCIFParser instance
        name: String. name of protein
        file_path: String. Filpath of the pdb or cif file to be read.

    Return:
        a Bio.PDB.Structure object representing the protein structure.
    """
    if file_path.rstrip(".gz").endswith("pdb"):
        return _parse_structure(pdb_parser, name, file_path)
    else:
        return _parse_structure(cif_parser, name, file_path)


def get_energy(pdb_file_path):
    """Get total pose energy from a PDB file.

    Args:
        pdb_file_path: String. Path to the pdb file.

    Return:
        Energy score.

    """
    if pdb_file_path.endswith(".pdb.gz"):
        pdb_file = gunzip_to_ram(pdb_file_path)
    else:
        pdb_file = open(pdb_file_path, "r")

    parse = False
    score = None
    for line in pdb_file:
        if line.startswith("#BEGIN_POSE_ENERGIES_TABLE"):
            parse = True
        if parse and line.startswith("pose"):
            score = float(line.strip().split()[-1])
    return score


def three_to_one_standard(res):
    """Encode non-standard AA to X.

    Args:
        res: a Bio.PDB.Residue object representing the residue.

    Return:
        String. One letter code of the residue.
    """
    if not is_aa(res, standard=True):
        return "X"
    return three_to_one(res)


def is_aa_by_target_atoms(res):
    """Tell if a Residue object is AA

    Args:
        res: a Bio.PDB.Residue object representing the residue.

    Return:
       Bool. Wheather or not the residue is AA.
    """
    target_atoms = ["N", "CA", "C", "O"]
    for atom in target_atoms:
        try:
            res[atom]
        except KeyError:
            return False
    return True


def get_atom_coords(residue, target_atoms=["N", "CA", "C", "O"]):
    """Extract the coordinates of the target_atoms from an AA residue.
    Handles exception where residue doesn't contain certain atoms
    by setting coordinates to np.nan

    Args:
        residue: a Bio.PDB.Residue object.
        target_atoms: Target atoms which residues will be resturned.

    Returns:
        np arrays with target atoms 3D coordinates in the order of target atoms.
    """
    atom_coords = []
    for atom in target_atoms:
        try:
            coord = residue[atom].coord
        except KeyError:
            coord = [np.nan] * 3
        atom_coords.append(coord)
    return np.asarray(atom_coords)


def chain_to_coords(
    chain, target_atoms=["N", "CA", "C", "O"], name="", residue_smiles=False
):
    """Convert a PDB chain in to coordinates of target atoms from all
    AAs

    Args:
        chain: a Bio.PDB.Chain object
        target_atoms: Target atoms which residues will be resturned.
        name: String. Name of the protein.
        residue_smiles: bool. Whether to get a list of smiles strings for the residues
    Returns:
        Dictonary containing protein sequence `seq`, 3D coordinates `coord` and name `name`.

    """
    output = {}
    # get AA sequence in the pdb structure
    pdb_seq = "".join(
        [
            three_to_one_standard(res.get_resname())
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    if len(pdb_seq) <= 1:
        # has no or only 1 AA in the chain
        return None
    output["seq"] = pdb_seq
    if residue_smiles:
        residues = []
        for res in chain.get_residues():
            if is_aa(res):
                mol = residue_to_mol(res)
                residues.append(Chem.MolToSmiles(mol))
        output["residues"] = residues
    # get the atom coords
    coords = np.asarray(
        [
            get_atom_coords(res, target_atoms=target_atoms)
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    output["coords"] = coords.tolist()
    output["name"] = "{}-{}".format(name, chain.id)
    return output


def read_file_from_s3(bucket: str, prefix: str):
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket, prefix)
    return obj.get()["Body"]


def extract_coords(
    structure, target_atoms=["N", "CA", "C", "O"], residue_smiles=False
):
    """
    Extract the atomic coordinates for all the chains.
    """
    records = {}
    for chain in structure.get_chains():
        record = chain_to_coords(
            chain,
            name=structure.id,
            target_atoms=target_atoms,
            residue_smiles=residue_smiles,
        )
        if record is not None:
            records[chain.id] = record
    return records


def parse_pdb_ids(pdb_ids: list, residue_smiles=False) -> dict:
    """
    Parse a list of PDB ids to structures by first retrieving
    PDB files from AWS OpenData Registry, then parse to structure objects.
    """
    PDB_BUCKET_NAME = "pdbsnapshots"
    pdb_parser = PDBParser(
        QUIET=True,
        PERMISSIVE=True,
        structure_builder=SloppyStructureBuilder(),
    )
    parsed_structures = {}
    for pdb_id in tqdm(pdb_ids):
        try:
            pdb_file = read_file_from_s3(
                PDB_BUCKET_NAME,
                f"20220103/pub/pdb/data/structures/all/pdb/pdb{pdb_id.lower()}.ent.gz",
            )
        except Exception as e:
            print(pdb_id, "caused the following error:")
            print(e)
        else:
            structure = pdb_parser.get_structure(
                pdb_id, gunzip_to_ram(pdb_file)
            )
            rec = extract_coords(structure, residue_smiles=residue_smiles)
            parsed_structures[pdb_id] = rec
    return parsed_structures


def remove_nan_residues(rec: dict) -> dict:
    """
    Remove the residues from a parsed protein chain where coordinates contains nan's
    """
    if len(rec["coords"]) == 0:
        return None
    coords = np.asarray(rec["coords"])  # shape: (n_residues, 4, 3)
    mask = np.isfinite(coords.sum(axis=(1, 2)))
    if mask.sum() == 0:
        # all residues coordinates are nan's
        return None
    if mask.sum() < coords.shape[0]:
        rec["seq"] = "".join(np.asarray(list(rec["seq"]))[mask])
        rec["coords"] = coords[mask].tolist()
    return rec


def residue_to_mol(residue: PDBEntity, **kwargs) -> Chem.rdchem.Mol:
    """Convert a parsed Biopython PDB object (Residue, Chain, Structure) to a
    rdkit Mol object"""
    # Write the PDB object into PDB string
    stream = StringIO()
    pdbio = PDBIO()
    pdbio.set_structure(residue)
    pdbio.save(stream)
    # Parse the PDB string with rdkit
    mol = Chem.MolFromPDBBlock(stream.getvalue(), **kwargs)
    return mol


def mol_to_pdb_structure(
    mol: Chem.rdchem.Mol, pdb_parser=None, protein_id=""
) -> PDBEntity:
    """
    Convert a rdkit Mol object to a Biopython PDB Structure object
    """
    # Write the Mol object into PDB string
    stream = StringIO()
    stream.write(Chem.MolToPDBBlock(mol))
    stream.seek(0)
    # parse the stream into a PDB Structure object
    if not pdb_parser:
        pdb_parser = PDBParser(
            QUIET=True,
            PERMISSIVE=True,
            structure_builder=SloppyStructureBuilder(),
        )
    structure = pdb_parser.get_structure(protein_id, stream)
    return structure
