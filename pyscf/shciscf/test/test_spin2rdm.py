import os
import numpy as np
import unittest

from pyscf import gto, scf, mcscf, lib
from pyscf.shciscf import shci
from pyscf.shciscf import spin_utils

# Test whether SHCI executable can be found. If it can, trigger tests that
# require it.
NO_SHCI = True
if shci.settings.SHCIEXE != 'shci_emulator' and shci.settings.SHCIEXE != None:
    print("Found SHCI =>", shci.settings.SHCIEXE)
    NO_SHCI = False
else:
    print("No SHCI found")

npt = np.testing

#
# Helper functions
#

def print_test(mc: mcscf.CASSCF, test_name: str, error: float, tol: float = 1e-12):
    """Testing utility"""
    if error > tol:
        lib.logger.note(mc.mol, f"\t{test_name} error {error:.3e} > {tol:.1e}")
        lib.logger.note(mc.mol, f"\t\033[91mFailed\033[00m {test_name} Test....")
        return 1
    else:
        lib.logger.note(mc.mol, f"\t\033[92mPassed\033[00m {test_name} Test....")
        return 0


def run_spin_test(mol: gto.mole, ncas: int, nelecas: int, nroots: int, name: str):
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff.copy()

    trusted_mc = mcscf.CASCI(mf, ncas, nelecas)
    trusted_mc.fcisolver.nroots = nroots
    trusted_mc.kernel(mo)

    mf.mo_coeff = mo

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = shci.SHCI(mol)
    mc.fcisolver.sweep_iter = [0, 3, 6]
    mc.fcisolver.sweep_epsilon = [1e-10] * 3
    mc.fcisolver.nroots = nroots
    mc.fcisolver.scratchDirectory = "."
    mc.kernel(mo)

    if isinstance(mc.nelecas, int):
        nelec = (mc.nelecas // 2, mc.nelecas // 2)
    else:
        nelec = mc.nelecas


    # Test <S^2> for each root
    for ni in range(nroots):
        lib.logger.note(mol, "##############")
        lib.logger.note(mol, f"#   ROOT {ni}   #")
        lib.logger.note(mol, "##############")
        (dice_dm1a, dice_dm1b), (
            dice_dm2aa,
            dice_dm2ab,
            dice_dm2bb,
        ) = spin_utils.read_dice_spin_2rdm(mc.fcisolver, ni, ncas, nelec)
        (pyscf_dm1a, pyscf_dm1b), (
            pyscf_dm2aa,
            pyscf_dm2ab,
            pyscf_dm2bb,
        ) = trusted_mc.fcisolver.make_rdm12s(
            trusted_mc.ci[ni], trusted_mc.ncas, trusted_mc.nelecas
        )

        # Testing
        dm1a_err = np.linalg.norm(dice_dm1a - pyscf_dm1a) / dice_dm1a.size
        dm1b_err = np.linalg.norm(dice_dm1b - pyscf_dm1b) / dice_dm1b.size
        dm2aa_err = np.linalg.norm(dice_dm2aa - pyscf_dm2aa) / dice_dm2aa.size
        dm2ab_err = np.linalg.norm(dice_dm2ab - pyscf_dm2ab) / dice_dm2ab.size
        dm2bb_err = np.linalg.norm(dice_dm2bb - pyscf_dm2bb) / dice_dm2bb.size

        npt.assert_almost_equal(dm1a_err, 0.0, decimal=5, err_msg=f"{name} root={ni} DM1a")
        npt.assert_almost_equal(dm1b_err, 0.0, decimal=5, err_msg=f"{name} root={ni} DM1b")
        npt.assert_almost_equal(dm2aa_err, 0.0, decimal=5, err_msg=f"{name} root={ni} DM2aa")
        npt.assert_almost_equal(dm2ab_err, 0.0, decimal=5, err_msg=f"{name} root={ni} DM2ab")
        npt.assert_almost_equal(dm2bb_err, 0.0, decimal=5, err_msg=f"{name} root={ni} DM2bb")

        pyscf_ss = trusted_mc.fcisolver.spin_square(trusted_mc.ci[ni], trusted_mc.ncas, nelec)[0]
        dice_ss = mc.fcisolver.spin_square(ni, ncas, nelec)[0]
        npt.assert_almost_equal(dice_ss, pyscf_ss, decimal=7, err_msg=f"{name} root={ni} <S^2>")

    mc.fcisolver.cleanup_dice_files()
    os.system("rm -f *.bkp")
    os.system("rm -f ./*RDM*.txt")


#
# The tests
#
class KnownValues(unittest.TestCase):
    @unittest.skipIf(NO_SHCI, "No SHCI Settings Found")
    def test_spin2RDM_C2(self):
        ncas, nelecas= (8, 8)
        spin= 0
        nroots = 9
        atom = "C 0 0 0;C 0 0 1"
        name = "C2"
        verbose = 4
        mol = gto.M(
            atom=atom,
            basis="ccpvdz",
            verbose=verbose,
            spin=spin,
            symmetry=True,
            output=f"{name}.out",
        )
        run_spin_test(mol, ncas, nelecas, nroots, name)

    @unittest.skipIf(NO_SHCI, "No SHCI Settings Found")
    def test_spin2RDM_CN(self):
        ncas, nelecas= (8, 9)
        spin= 1
        nroots = 4
        atom = "C 0 0 0; N 0 0 1"
        name = "CN"
        verbose = 4

        mol = gto.M(
            atom=atom,
            basis="ccpvdz",
            verbose=verbose,
            spin=spin,
            symmetry=True,
            output=f"{name}.out",
        )
        run_spin_test(mol, ncas, nelecas, nroots, name)


    @unittest.skipIf(NO_SHCI, "No SHCI Settings Found")
    def test_spin2RDM_O2(self):
        ncas, nelecas= (8, 12)
        spin= 0
        nroots = 4
        atom = "O 0 0 0; O 0 0 1.208"
        name = "O2"
        verbose = 4

        mol = gto.M(
            atom=atom,
            basis="ccpvdz",
            verbose=verbose,
            spin=spin,
            symmetry=True,
            output=f"{name}.out",
        )
        run_spin_test(mol, ncas, nelecas, nroots, name)

if __name__ == "__main__":
    print("Tests for shciscf interface")
    unittest.main()