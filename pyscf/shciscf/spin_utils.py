import sys, os, ctypes
import numpy as np
from pyscf import gto, scf, mcscf, lib
from pyscf.lib import load_library

# TODO: Organize this better.
shciLib = load_library("libshciscf")
ndpointer = np.ctypeslib.ndpointer

r2RDM = shciLib.r2RDM
r2RDM.restype = None
r2RDM.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ctypes.c_char_p,
]


def read_Dice2RDM(dice2RDMName):
    with open(dice2RDMName) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    dice2RDM = np.zeros((norbs,) * 4)

    for i in range(1, len(content)):
        c0, c1, d1, d0, val = content[i].split()
        dice2RDM[int(c0), int(c1), int(d1), int(d0)] = float(val)

    return dice2RDM

def read_dice_spin_2rdm(fcisolver, root : int, ncas: int, nelec : tuple):
    n_spin_orbs = ncas * 2 
    spin_2rdm = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs), order="C")
    filename = os.path.join(fcisolver.scratchDirectory,"spin2RDM.%d.%d.txt" % (root, root))
    r2RDM(spin_2rdm, n_spin_orbs, filename.encode())

    dm2aa = spin_2rdm[::2, ::2, ::2, ::2] #.transpose(0, 2, 1, 3)
    dm2ab = spin_2rdm[::2, ::2, 1::2, 1::2] #.transpose(0, 2, 1, 3)
    dm2bb = spin_2rdm[1::2, 1::2, 1::2, 1::2] #.transpose(0, 2, 1, 3)

    # We're summing over alpha and beta so we need to divide by their sum
    # fmt: off
    dm1a = np.einsum("ikjj", spin_2rdm)[::2, ::2] / (nelec[0]+nelec[1] - 1.0)
    dm1b = np.einsum("jjik", spin_2rdm)[1::2, 1::2] / (nelec[0]+nelec[1] - 1.0)
    # fmt: on
    # print("dm1a nelec", np.einsum("ii", dm1a))
    # print("dm1b nelec", np.einsum("ii", dm1b))
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def restricted_spin_square(fcisolver, root: int, ncas: int, nelecas: int) -> float:
    """Calculate <S^2> for an SHCI object with restricted orbitals.

    If this function can't find spin2RDM files from Dice, it will return
    approximate (and possibly incorrect) values for <S^2> and mutliplicity.
    In this case, it calculates <S^2> as <S_z>^2.

    Parameters
    ----------
    fcisolver :
        An fcisolver-type object, intended for use with SHCI.
    root : int
        The root we want <S^2>, use 0-based indexing.
    ncas : int
        The number of active space orbitals.
    nelec : tuple
        Specify the desired nelec tuple, e.g. (8,6). By default None

    Returns
    -------
    (float, float)
        <S^2> for the root of the system in question at index 0 and the 
        multiplicity at index 1.

    """

    lib.logger.debug(fcisolver.mol, f"\tCalculating <S^2> for CI-type |\u03A8> root: {root}")

    # Input checking
    nelec = nelecas

    if os.path.exists(os.path.join(fcisolver.scratchDirectory,"spin2RDM.%d.%d.txt" % (root, root))):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = read_dice_spin_2rdm(fcisolver, root, ncas, nelec)
    else:
        lib.logger.warn(fcisolver.mol, f"\tNO spin2RDM found from Dice, make sure you're using the latest version")
        lib.logger.warn(fcisolver.mol, f"\tSetting <S^2> = <S_Z>^2")
        s_z = (nelec[0] - nelec[1]) * 0.5
        ss = s_z * (s_z + 1)
        return ss, s_z * 2 + 1

    # Actual Calculation
    ovlpaa = ovlpab = ovlpba = ovlpbb = np.diag(np.ones(ncas))

    # Make sure <S_z^2> from the RDMs matches ((alpha-beta)/2.0)**2
    # if ovlp=1, ssz = (neleca-nelecb)**2 * .25
    ssz = (
        np.einsum("ijkl,ij,kl->", dm2aa, ovlpaa, ovlpaa)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
        + np.einsum("ijkl,ij,kl->", dm2bb, ovlpbb, ovlpbb)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
    ) * 0.25
    ssz += (
        np.einsum("ji,ij->", dm1a, ovlpaa) + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.25
    # End testing

    dm2abba = -dm2ab.transpose(0, 3, 2, 1)  # alpha^+ beta^+ alpha beta
    dm2baab = -dm2ab.transpose(2, 1, 0, 3)  # beta^+ alpha^+ beta alpha

    # Calculate contribution from S_- and S_+
    ssxy = (
        np.einsum("ijkl,ij,kl->", dm2baab, ovlpba, ovlpab)
        + np.einsum("ijkl,ij,kl->", dm2abba, ovlpab, ovlpba)
        + np.einsum("ji,ij->", dm1a, ovlpaa)
        + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.5

    sz = (nelec[0] - nelec[1]) / 2.0
    szsz = sz * sz
    ss = szsz + ssxy
    

    # Warn of if things aren't as we'd expect
    lib.logger.debug(fcisolver.mol, f"\t<S_z> numerical error {abs(szsz - ssz):.3e}")
    lib.logger.debug(
        fcisolver.mol, f"\t<S^2> = {ss:.3f}    <S_z^2> = {szsz:.3f}    <S_+S_-> = {ssxy:.3f}"
    )

    # Calculate <S>
    s = np.sqrt(ss + 0.25) - 0.5

    return ss, s * 2.0 + 1.0