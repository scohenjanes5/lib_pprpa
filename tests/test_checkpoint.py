import h5py
import numpy as np

from lib_pprpa import pyscf_util
from lib_pprpa.grad.ase_utils import pprpaobj
from lib_pprpa.pprpa_davidson import get_identity_trial_vector
from test_utils import get_water_rks


def test_checkpoint_truncates_and_restarts(tmp_path):
    # Run a small ppRPA job that writes a checkpoint and ensure only the
    # populated trial vectors are stored (no padded zeros), then restart from it.
    mol, mf = get_water_rks(df=True)
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol(mf)

    chkfile = tmp_path / "pprpa_chk.h5"
    pp = pprpaobj(mf, "pp", Lpq=Lpq, nroot=1, checkpoint=str(chkfile))
    pp.max_vec = 500  # ensure the saved slice would be much smaller than max_vec
    pp.kernel("s")
    exci_saved = pp.exci_s.copy()

    with h5py.File(chkfile, "r") as f:
        g = f["singlet"]
        ntri = int(np.asarray(g["ntri"]))
        tri_shape = g["tri_vec"].shape
        sig_shape = g["tri_vec_sig"].shape

        assert tri_shape[0] == ntri
        assert sig_shape[0] == ntri
        assert ntri < pp.max_vec  # confirms we did not save padded rows

    # Restart from the checkpoint to verify the truncated data is loadable
    pp_restart = pprpaobj(mf, "pp", Lpq=Lpq, nroot=1, checkpoint=str(chkfile))
    pp_restart.max_vec = 500
    pp_restart.kernel("s")
    # Restart should load the truncated trial vectors and produce the same result.
    assert np.allclose(pp_restart.exci_s, exci_saved)


def test_checkpoint_restart_expands_and_grows_workspace(tmp_path):
    # Build a tiny checkpoint with only a couple of trial vectors so restart must expand.
    mol, mf = get_water_rks(df=True)
    _, _, Lpq = pyscf_util.get_pyscf_input_mol(mf)

    chkfile = tmp_path / "pprpa_chk_expand.h5"

    # Reference result from a clean run for comparison
    clean = pprpaobj(mf, "pp", Lpq=Lpq, nroot=1, checkpoint=None)
    clean.max_vec = 20
    clean.kernel("s")
    exci_ref = clean.exci_s.copy()

    # Manually create a truncated checkpoint with a very small subspace (ntri=2)
    pp = pprpaobj(mf, "pp", Lpq=Lpq, nroot=1, checkpoint=str(chkfile))
    pp.max_vec = 20
    pp.multi = "s"
    pp.check_parameter()
    tri_vec, tri_sig = get_identity_trial_vector(pp, ntri=2)
    mv_prod = pp.contraction(tri_vec)
    pp._save_pprpa_checkpoint(conv=False, ntri=2, tri_vec=tri_vec, tri_vec_sig=tri_sig)

    # Restart should load the tiny checkpoint, expand beyond ntri=2, and converge
    pp_restart = pprpaobj(mf, "pp", Lpq=Lpq, nroot=1, checkpoint=str(chkfile))
    pp_restart.max_vec = 20
    pp_restart.kernel("s")

    # Ensure expansion happened (final saved ntri grew beyond the initial 2)
    with h5py.File(chkfile, "r") as f:
        ntri_final = int(np.asarray(f["singlet"]["ntri"]))
    assert ntri_final > 2

    # Final energies should match the clean run
    assert np.allclose(pp_restart.exci_s, exci_ref)
