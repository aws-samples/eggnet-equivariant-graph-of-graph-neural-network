"""
Utilities for computing physically relevant properties of molecules
Source: https://github.com/ACE-KAIST/PIGNet/blob/main/dataset.py
"""

import pickle
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Atom, Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")
random.seed(0)

INTERACTION_TYPES = [
    # "saltbridge",
    "hbonds",
    # "pication",
    # "pistack",
    # "halogen",
    # "waterbridge",
    "hydrophobic",
    "metal_complexes",
]
pt = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""
PERIODIC_TABLE = dict()
for i, per in enumerate(pt.split()):
    for j, ele in enumerate(per.split(",")):
        PERIODIC_TABLE[ele] = (i, j)
PERIODS = [0, 1, 2, 3, 4, 5]
GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
FORMALCHARGES = [-2, -1, 0, 1, 2, 3, 4]
METALS = ("Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu")
HYDROPHOBICS = ("F", "CL", "BR", "I")
VDWRADII = {
    6: 1.90,
    7: 1.8,
    8: 1.7,
    16: 2.0,
    15: 2.1,
    9: 1.5,
    17: 1.8,
    35: 2.0,
    53: 2.2,
    30: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    12: 1.2,
    28: 1.2,
    20: 1.2,
    29: 1.2,
}
HBOND_DONOR_INDICES = ["[!#6;!H0]"]
HBOND_ACCEPPTOR_SMARTS = [
    "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]


def get_period_group(atom: Atom) -> List[bool]:
    period, group = PERIODIC_TABLE[atom.GetSymbol().upper()]
    return one_of_k_encoding(period, PERIODS) + one_of_k_encoding(
        group, GROUPS
    )


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set)
        )
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: Any, allowable_set: List[Any]) -> List[bool]:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(mol: Mol, atom_index: int) -> np.ndarray:
    atom = mol.GetAtomWithIdx(atom_index)
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), DEGREES)
        + one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + one_of_k_encoding_unk(atom.GetFormalCharge(), FORMALCHARGES)
        + get_period_group(atom)
        + [atom.GetIsAromatic()]
    )  # (9, 6, 7, 7, 24, 1) --> total 54


def get_atom_feature(mol: Mol) -> np.ndarray:
    natoms = mol.GetNumAtoms()
    H = []
    for idx in range(natoms):
        H.append(atom_feature(mol, idx))
    H = np.array(H)
    return H


def get_vdw_radius(atom: Atom) -> float:
    atomic_num = atom.GetAtomicNum()
    if VDWRADII.get(atomic_num):
        return VDWRADII[atomic_num]
    return Chem.GetPeriodicTable().GetRvdw(atomic_num)


def get_hydrophobic_atom(mol: Mol) -> np.ndarray:
    natoms = mol.GetNumAtoms()
    hydrophobic_indice = np.zeros((natoms,))
    for atom_idx in range(natoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol.upper() in HYDROPHOBICS:
            hydrophobic_indice[atom_idx] = 1
        elif symbol.upper() in ["C"]:
            neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
            neighbors_wo_c = list(set(neighbors) - set(["C"]))
            if len(neighbors_wo_c) == 0:
                hydrophobic_indice[atom_idx] = 1
    return hydrophobic_indice


def get_A_hydrophobic(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_indice = get_hydrophobic_atom(ligand_mol)
    target_indice = get_hydrophobic_atom(target_mol)
    return np.outer(ligand_indice, target_indice)


def get_hbond_atom_indices(mol: Mol, smarts_list: List[str]) -> np.ndarray:
    indice = []
    for smarts in smarts_list:
        smarts = Chem.MolFromSmarts(smarts)
        indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
    indice = np.array(indice)
    return indice


def get_A_hbond(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_ACCEPPTOR_SMARTS
    )
    target_h_acc_indice = get_hbond_atom_indices(
        target_mol, HBOND_ACCEPPTOR_SMARTS
    )
    ligand_h_donor_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_DONOR_INDICES
    )
    target_h_donor_indice = get_hbond_atom_indices(
        target_mol, HBOND_DONOR_INDICES
    )

    hbond_indice = np.zeros(
        (ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
    )
    for i in ligand_h_acc_indice:
        for j in target_h_donor_indice:
            hbond_indice[i, j] = 1
    for i in ligand_h_donor_indice:
        for j in target_h_acc_indice:
            hbond_indice[i, j] = 1
    return hbond_indice


def get_A_metal_complexes(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(
        ligand_mol, HBOND_ACCEPPTOR_SMARTS
    )
    target_h_acc_indice = get_hbond_atom_indices(
        target_mol, HBOND_ACCEPPTOR_SMARTS
    )
    ligand_metal_indice = np.array(
        [
            idx
            for idx in range(ligand_mol.GetNumAtoms())
            if ligand_mol.GetAtomWithIdx(i).GetSymbol() in METALS
        ]
    )
    target_metal_indice = np.array(
        [
            idx
            for idx in range(target_mol.GetNumAtoms())
            if target_mol.GetAtomWithIdx(i).GetSymbol() in METALS
        ]
    )

    metal_indice = np.zeros(
        (ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
    )
    for ligand_idx in ligand_h_acc_indice:
        for target_idx in target_metal_indice:
            metal_indice[ligand_idx, target_idx] = 1
    for ligand_idx in ligand_metal_indice:
        for target_idx in target_h_acc_indice:
            metal_indice[ligand_idx, target_idx] = 1
    return metal_indice


def get_interaction_indices(ligand_mol: Mol, target_mol: Mol) -> np.array:
    interaction_indice = np.zeros(
        (
            len(INTERACTION_TYPES),
            ligand_mol.GetNumAtoms(),
            target_mol.GetNumAtoms(),
        )
    )
    interaction_indice[0] = get_A_hbond(ligand_mol, target_mol)
    interaction_indice[1] = get_A_metal_complexes(ligand_mol, target_mol)
    interaction_indice[2] = get_A_hydrophobic(ligand_mol, target_mol)
    return interaction_indice


def mol_to_feature(
    ligand_mol: Mol, target_mol: Mol, compute_full: bool = False
) -> Dict[str, Any]:
    """
    Args:
        compute_full: if True, compute components for
            both inter and intra energies.
    """
    # Remove hydrogens
    ligand_mol = Chem.RemoveHs(ligand_mol, sanitize=False)
    target_mol = Chem.RemoveHs(target_mol, sanitize=False)

    # prepare ligand
    ligand_pos = np.array(ligand_mol.GetConformers()[0].GetPositions())

    # prepare protein
    target_pos = np.array(target_mol.GetConformers()[0].GetPositions())

    interaction_indice = get_interaction_indices(ligand_mol, target_mol)

    # count rotatable bonds
    rotor = CalcNumRotatableBonds(ligand_mol)

    # no metal
    ligand_non_metal = np.array(
        [
            1 if atom.GetSymbol() not in METALS else 0
            for atom in ligand_mol.GetAtoms()
        ]
    )
    target_non_metal = np.array(
        [
            1 if atom.GetSymbol() not in METALS else 0
            for atom in target_mol.GetAtoms()
        ]
    )
    # vdw radius
    ligand_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in ligand_mol.GetAtoms()]
    )
    target_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in target_mol.GetAtoms()]
    )

    sample = {
        "interaction_indice": interaction_indice,
        "ligand_pos": ligand_pos,
        "target_pos": target_pos,
        "rotor": rotor,
        "ligand_vdw_radii": ligand_vdw_radii,
        "target_vdw_radii": target_vdw_radii,
        "ligand_non_metal": ligand_non_metal,
        "target_non_metal": target_non_metal,
    }
    if compute_full:
        sample["ligand_interaction_indice"] = get_interaction_indices(
            ligand_mol, ligand_mol
        )
        sample["target_interaction_indice"] = get_interaction_indices(
            target_mol, target_mol
        )
        sample["rotor_target"] = CalcNumRotatableBonds(target_mol)
    return sample
