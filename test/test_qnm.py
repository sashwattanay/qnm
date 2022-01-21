import pytest
import qnm
import numpy as np
try:
    from pathlib import Path # py 3
except ImportError:
    from pathlib2 import Path # py 2








class QnmTestDownload(object):
    """
    Base class so that each test will automatically download_data
    """
    @classmethod
    def setup_class(cls):
        """
        Download the data when setting up the test class.
        """
        qnm.download_data()

class TestQnmFileOps(QnmTestDownload):
    def test_cache_file_operations(self):
        """Test file operations and downloading the on-disk cache.
        """

        print("Downloading with overwrite=True")
        qnm.cached.download_data(overwrite=True)
        print("Clearing disk cache but not tarball")
        qnm.cached._clear_disk_cache(delete_tarball=False)
        print("Decompressing tarball")
        qnm.cached._decompress_data()

class TestQnmOneMode(QnmTestDownload):
    def test_one_mode(self):
        """
        An example of a test
        """
        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        omega, A, C = grav_220(a=0.68)
        assert np.allclose(omega, (0.5239751042900845 - 0.08151262363119974j))














class TestSepConstDerivative(QnmTestDownload):
    @pytest.mark.parametrize("A0, s, c, m, l_max",
                             [(1.0, 2, 0.1, 2, 5),      # Low spin
                              (1.0, 2, 0.9, 2, 5),      # High spin
                              (1.0, 2, 0.5, 2, 8),      # High spin
                              (1.0, 1, 0.5, 2, 8),      # High spin
                              (1.0, 2, 0.5, 2, 5),      # High spin
                              (1.0, -1, 0.5, 2, 5),      # High spin
                              ])

    def test_sep_const_derivative(self, A0, s, c, m, l_max):

        from qnm.angular import C_and_sep_const_closest_and_deriv_of_sep_const

        """
        See if the serivative of the separation constant (eigenvalue) is correctly given by
        the code by comparing it with the finite difference method
        """

        e = 1.0e-3

        dc = e*complex(1,0)
        C0    =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c, m, l_max)[0]
        Cfwd  =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c+dc, m, l_max)[0]
        Cfwd2 =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c+2*dc, m, l_max)[0]
        Cbwd  =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c-dc, m, l_max)[0]
        Cbwd2 =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c-2*dc, m, l_max)[0]
        C_der_fin_diff_real = (-Cfwd2 + 8* Cfwd - 8*Cbwd + Cbwd2)/(12*dc)
        C_der_analytic_real = C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c, m, l_max)[2]

        dc = e*complex(0,1)
        C0    =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c, m, l_max)[0]
        Cfwd  =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c+dc, m, l_max)[0]
        Cfwd2 =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c+2*dc, m, l_max)[0]
        Cbwd  =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c-dc, m, l_max)[0]
        Cbwd2 =  C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c-2*dc, m, l_max)[0]
        C_der_fin_diff_imag = (-Cfwd2 + 8* Cfwd - 8*Cbwd + Cbwd2)/(12*dc)
        C_der_analytic_imag = C_and_sep_const_closest_and_deriv_of_sep_const(A0, s, c, m, l_max)[2]

        assert np.allclose([C_der_fin_diff_real, C_der_fin_diff_imag], [C_der_analytic_real, C_der_analytic_imag])



class TestQnmNewLeaverSolver(QnmTestDownload):
    def test_compare_old_new_Leaver(self):
        """ Check consistency between old and new Leaver solvers """
        from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
        old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
        assert np.all([old[i] == new[i] for i in range(3)])

class TestQnmSolveInterface(QnmTestDownload):
    """
    Test the various interface options for solving
    """

    def test_interp_only(self):
        """Check that we get reasonable values (but not identical!)
        with just interpolation.
        """

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        a = 0.68
        assert a not in grav_220.a

        omega_int, A_int, C_int = grav_220(a=a, interp_only=True)
        omega_sol, A_sol, C_sol = grav_220(a=a, interp_only=False, store=False)

        assert np.allclose(omega_int, omega_sol) and not np.equal(omega_int, omega_sol)
        assert np.allclose(A_int, A_sol) and not np.equal(A_int, A_sol)
        assert np.allclose(C_int, C_sol) and not all(np.equal(C_int, C_sol))

    def test_store_a(self):
        """Check that the option store=True updates a spin sequence"""

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)

        old_n = len(grav_220.a)
        k = int(old_n/2)

        new_a = 0.5 * (grav_220.a[k] + grav_220.a[k+1])

        assert new_a not in grav_220.a

        _, _, _ = grav_220(new_a, store=False)
        n_1 = len(grav_220.a)
        assert old_n == n_1

        _, _, _ = grav_220(new_a, store=True)
        n_2 = len(grav_220.a)
        assert n_2 == n_1 + 1

    def test_resolve(self):
        """Test that option resolve_if_found=True really does a new
        solve"""

        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)

        n = len(grav_220.a)
        k = int(n/2)
        a = grav_220.a[k]

        grav_220.solver.solved = False
        omega_old, A_old, C_old = grav_220(a=a, resolve_if_found=False)
        solved_1 = grav_220.solver.solved

        omega_new, A_new, C_new = grav_220(a=a, resolve_if_found=True)
        solved_2 = grav_220.solver.solved

        assert (solved_1 is False) and (solved_2 is True)
        assert np.allclose(omega_new, omega_old)
        assert np.allclose(A_new, A_old)
        assert np.allclose(C_new, C_old)

class TestMirrorModeTransformation(QnmTestDownload):
    @pytest.mark.parametrize(  "s, l, m, n, a",
                             [(-2, 2, 2, 0, 0.1),  # Low spin
                              (-2, 2, 2, 0, 0.9),  # High spin
                              (-2, 2, 2, 4, 0.7),  # Different overtone
                              (-2, 3, 2, 0, 0.7),  # l odd
                              (-2, 3, 1, 0, 0.7),  # l and m odd
                              (-1, 3, 1, 0, 0.7),  # s, l, and m odd
                              ])
    def test_mirror_mode_transformation(self, s, l, m, n, a):
        import copy

        mode = qnm.modes_cache(s=s, l=l, m=m, n=n)
        om, A, C = mode(a=a)

        solver = copy.deepcopy(mode.solver) # need to import copy -- don't want to actually modify this mode's solver
        solver.clear_results()
        solver.set_params(a=a, m=-m, A_closest_to=A.conj(), omega_guess=-om.conj())
        om_prime = solver.do_solve()

        assert np.allclose(-om.conj() , solver.omega)
        assert np.allclose(A.conj(), solver.A)
        assert np.allclose((-1)**(l + qnm.angular.ells(s, m, mode.l_max)) * C.conj(), solver.C)

@pytest.mark.slow
class TestQnmBuildCache(QnmTestDownload):
    def test_build_cache(self):
        """Check the default cache-building functionality"""

        qnm.cached._clear_disk_cache(delete_tarball=False)
        qnm.modes_cache.seq_dict = {}
        qnm.cached.build_package_default_cache(qnm.modes_cache)
        assert 860 == len(qnm.modes_cache.seq_dict.keys())
        qnm.modes_cache.write_all()
        cache_data_dir = qnm.cached.get_cachedir() / 'data'

        # Magic number, default num modes is 860
        assert 860 == len(list(cache_data_dir.glob('*.pickle')))
