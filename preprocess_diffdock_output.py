# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
# Convert DiffDock's output into the same format with CASF-2016 Docking data.

The output from DiffDock is a PDB file, which contains a single chain
representing the pose of the small molecule ligand.

Example usage:
python preprocess_diffdock_output.py \
    --data_dir /home/ec2-user/SageMaker/efs/data/DiffDockData/inference1 \
    --pdb_data_dir /home/ec2-user/SageMaker/efs/data/DiffDockData/PDBBind_processed \
    --thres 6 \
    --output_dir /home/ec2-user/SageMaker/efs/data/DiffDockData/inference1_processed_t6

In this pipeline, we perform the following steps to convert the outputs from
DiffDock to a format compatible with our affinity prediction model:

1. parse the ligand pose PDB file
2. combine the ligand with protein PDB into the same coordinate system
3. subset the protein chain(s) to only include the residues around the ligand
    -> pocket-ligand structure
4. save the pocket-ligand structures into pickles; save the RMSD values into
    text files

After these steps, we should be able to run inference using our affinity
prediction model in a similar settings in `evaluate_casf2016.py` function
`evaluate_docking`.
"""
import os
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser, Select, PDBIO
from Bio.PDB.Polypeptide import is_aa

from ppi.data_utils import parse_pdb_structure


def get_calpha_coords(residue):
    try:
        return residue["CA"].coord
    except KeyError:
        return [np.nan] * 3


def get_contact_residues(ligand_mol, chain, thres=10):
    # get the residue IDs
    res_ids = np.asarray(
        [res.id[1] for res in chain.get_residues() if is_aa(res)]
    )
    # ligand coords
    coords1 = ligand_mol.GetConformers()[0].GetPositions()
    # extract the C-alpha coordinates of all AA residues
    coords2 = np.asarray(
        [get_calpha_coords(res) for res in chain.get_residues() if is_aa(res)]
    )
    # calculate interchain distance
    dist = cdist(coords1, coords2)
    dist_bool = dist <= thres

    res_keep = res_ids[dist_bool.sum(axis=0) > 0]
    return res_keep


def get_contact_residues_across_chains(ligand_mol, protein, thres=10):
    d_chain_residues = {}
    for chain in protein.get_chains():
        res_keep = get_contact_residues(ligand_mol, chain, thres=thres)
        d_chain_residues[chain.id] = res_keep
    return d_chain_residues


def subset_pdb_structure(structure, d_chain_residues, outfile):
    # to subset the protein structure
    class ResSelect(Select):
        def accept_residue(self, res):
            if res.id[1] in d_chain_residues.get(res.parent.id, set()):
                return True
            else:
                return False

    io = PDBIO()
    # set the structure as the entire protein
    io.set_structure(structure)
    # subset and save the pocket into PDB file
    io.save(outfile, ResSelect())
    return


def process_one(row, pdb_parser, args):
    # 1. Parse ligand poses from PDB files
    ligand_mol = Chem.MolFromPDBFile(
        os.path.join(args.data_dir, row["pdb_file"]), sanitize=False
    )
    # 2. combine the ligand with protein PDB into the same coordinate system
    # parse the corresponding protein
    protein = parse_pdb_structure(
        pdb_parser,
        row.pdb_id,
        os.path.join(
            args.pdb_data_dir,
            row["pdb_id"],
            f"{row['pdb_id']}_protein_processed.pdb",
        ),
    )
    # 3. subset the protein chain(s) to only include the residues around the
    # ligand
    d_chain_residues = get_contact_residues_across_chains(
        ligand_mol, protein, thres=args.thres
    )
    subset_pdb_structure(
        protein,
        d_chain_residues,
        os.path.join(args.output_dir, f"{row['pdb_id']}_chopped.pdb"),
    )
    protein_pocket = parse_pdb_structure(
        pdb_parser,
        row.pdb_id,
        os.path.join(args.output_dir, f"{row['pdb_id']}_chopped.pdb"),
    )
    # 4. write to pickle
    output = (ligand_mol, None, protein_pocket, None)
    output_file = os.path.join(args.output_dir, "data", row["file_id"])
    pickle.dump(output, open(output_file, "wb"))
    return


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sub_dirs = ["decoys_docking_rmsd", "data", "keys"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(args.output_dir, sub_dir), exist_ok=True)

    # 0. parse metadata from DiffDock output files
    meta_df = []
    for pdb_file in os.listdir(args.data_dir):
        if pdb_file.endswith(".pdb"):
            row = {
                "id": pdb_file[:-4],
                "pdb_file": pdb_file,
                "pdb_id": pdb_file.split("_")[0],
                "rank": int(pdb_file.split("_")[1]),
                "rmsd": float(pdb_file.split("_")[2]),
                "confidence": float(pdb_file.split("_")[3][:-4]),
            }
            row["file_id"] = row["pdb_id"] + "_" + row["id"].split("_")[1]
            meta_df.append(row)

    meta_df = pd.DataFrame(meta_df).set_index("id", verify_integrity=True)
    print(meta_df.shape)

    pdb_parser = PDBParser(
        QUIET=True,
        PERMISSIVE=True,
    )
    for _, row in tqdm(meta_df.iterrows(), total=meta_df.shape[0]):
        process_one(row, pdb_parser, args)

    # Write RMSD files
    for pdb_id, sub_df in meta_df.groupby("pdb_id"):
        out_rmsd_filename = f"{pdb_id}_rmsd.dat"
        sub_df[["file_id", "rmsd"]].to_csv(
            os.path.join(
                args.output_dir, "decoys_docking_rmsd", out_rmsd_filename
            ),
            sep="\t",
            index=False,
        )
    # Write keys and pdb_to_affinity.txt
    keys = list(meta_df["file_id"])
    pickle.dump(
        keys, open(os.path.join(args.output_dir, "keys/test_keys.pkl"), "wb")
    )
    pdb_to_affinity = meta_df[["file_id"]]
    pdb_to_affinity.loc[:, "affinity"] = 0
    pdb_to_affinity.to_csv(
        os.path.join(args.output_dir, "pdb_to_affinity.txt"),
        sep="\t",
        index=False,
        header=False,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to the output ligand poses in pdb files from DiffDock",
    )
    parser.add_argument(
        "--pdb_data_dir",
        type=str,
        required=True,
        help="Directory to oringal protein pdb files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--thres",
        type=int,
        required=True,
        default=10,
        help="Threshold for identifying contact residues",
    )

    args = parser.parse_args()
    main(args)
