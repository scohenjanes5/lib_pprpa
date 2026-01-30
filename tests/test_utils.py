import numpy as np
from pyscf import gto, scf, dft, pbc
from pyscf.pbc import dft as pbc_dft

def get_water_mol():
    mol = gto.Mole()
    mol.atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (0.0, 0.757, 0.587)],
                ["H", (0.0, -0.757, 0.587)]]
    mol.basis = "6-31g"
    mol.verbose = 0
    mol.build()
    return mol

def get_water_cell():
    cell = pbc.gto.Cell()
    cell.atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (0.0, 0.757, 0.587)],
                ["H", (0.0, -0.757, 0.587)]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 5
    cell.verbose = 0
    cell.build()
    return cell

def get_water_rks(df=True):
    mol = get_water_mol()
    mf = dft.RKS(mol, xc="B3LYP")
    if df:
        mf = mf.density_fit()
    mf.kernel()
    return mol, mf

def get_water_rhf(df=True):
    mol = get_water_mol()
    mf = scf.RHF(mol)
    if df:
        mf = mf.density_fit()
    mf.kernel()
    return mol, mf

def get_water_ghf(df=True):
    mol = get_water_mol()
    mol.spin = 2
    mol.build()
    mf = scf.GHF(mol)
    if df:
        mf = mf.density_fit()
    mf.kernel()
    return mol, mf

def get_water_pbc_rks():
    cell = get_water_cell()
    mf = pbc_dft.RKS(cell)
    mf.xc = 'pbe'
    mf.kernel()
    return cell, mf
