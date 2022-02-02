""" Solve the radial Teukolsky equation via Leaver's method.
"""

from __future__ import division, print_function, absolute_import

from numba import njit
import numpy as np

from .contfrac import lentz

# TODO some documentation here, better documentation throughout

@njit(cache=True)
def sing_pt_char_exps(omega, a, s, m):
    r""" Compute the three characteristic exponents of the singular points
    of the radial Teukolsky equation.

    We want ingoing at the outer horizon and outgoing at infinity. The
    choice of one of two possible characteristic exponents at the
    inner horizon doesn't affect the minimal solution in Leaver's
    method, so we just pick one. Thus our choices are, in the
    nomenclature of [1]_, :math:`(\zeta_+, \xi_-, \eta_+)`.

    Parameters
    ----------
    omega: complex
      The complex frequency in the ansatz for the solution of the
      radial Teukolsky equation.

    a: double
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    Returns
    -------
    (complex, complex, complex)
      :math:`(\zeta_+, \xi_-, \eta_+)`

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    root = np.sqrt(1. - a*a)
    r_p, r_m = 1. + root, 1. - root

    sigma_p = (2.*omega*r_p - m*a)/(2.*root)
    sigma_m = (2.*omega*r_m - m*a)/(2.*root)

    zeta = +1.j * omega        # This is the choice \zeta_+
    xi   = - s - 1.j * sigma_p # This is the choice \xi_-
    eta  = -1.j * sigma_m      # This is the choice \eta_+

    return zeta, xi, eta

@njit(cache=True)
def D_coeffs(omega, a, s, m, A):
    """ The D_0 through D_4 coefficients that enter into the radial
    infinite continued fraction, Eqs. (31) of [1]_ .


    Parameters
    ----------
    omega: complex
      The complex frequency in the ansatz for the solution of the
      radial Teukolsky equation.

    a: double
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    Returns
    -------
    array[5] of complex
      D_0 through D_4 .

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    zeta, xi, eta = sing_pt_char_exps(omega, a, s, m)

    root  = np.sqrt(1. - a*a)

    p     = root * zeta
    alpha = 1. + s + xi + eta - 2.*zeta + s # Because we took the root \zeta_+
    gamma = 1. + s + 2.*eta
    delta = 1. + s + 2.*xi
    sigma = (A + a*a*omega*omega - 8.*omega*omega
             + p * (2.*alpha + gamma - delta)
             + (1. + s - 0.5*(gamma + delta))
             * (s + 0.5*(gamma + delta)))

    D = [0.j] * 5
    D[0] = delta
    D[1] = 4.*p - 2.*alpha + gamma - delta - 2.
    D[2] = 2.*alpha - gamma + 2.
    D[3] = alpha*(4.*p - delta) - sigma
    D[4] = alpha*(alpha - gamma + 1.)

    return D

def leaver_cf_trunc_inversion(omega, a, s, m, A,
                              n_inv, N=300, r_N=1.):
    """ Legacy function.

    Approximate the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    N terms total for the approximation. This uses "bottom up"
    evaluation, and you can pass a seed value r_N to assume for
    the rest of the infinite fraction which has been truncated.
    The value returned is Eq. (44) of [1]_.

    Parameters
    ----------
    omega: complex
      The complex frequency for evaluating the infinite continued
      fraction.

    a: float
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    n_inv: int
      Inversion number for the infinite continued fraction. Finding
      the nth overtone is typically most stable when n_inv = n .

    N: int, optional [default: 300]
      The depth where the infinite continued fraction is truncated.

    r_N: float, optional [default: 1.]
      Value to assume for the rest of the infinite continued fraction
      past the point of truncation.

    Returns
    -------
    complex
      The nth inversion of the infinite continued fraction evaluated
      with these arguments.

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    n = np.arange(0, N+1)

    D = D_coeffs(omega, a, s, m, A)

    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    conv2 = -r_N # Is this sign correct?
    for i in range(N, n_inv, -1): # n_inv is not included
        conv2 = gamma[i] / (beta[i] - alpha[i] * conv2)

    return (beta[n_inv]
            - gamma[n_inv] * conv1
            - alpha[n_inv] * conv2)

# TODO possible choices for r_N: 0., 1., approximation using (34)-(38)

# Definitions for a_i, b_i for continued fraction below
# In defining the below a, b sequences, I have cleared a fraction
# compared to the usual way of writing the radial infinite
# continued fraction. The point of doing this was that so both
# terms, a(n) and b(n), tend to 1 as n goes to infinity. Further,
# We can analytically divide through by n in the numerator and
# denominator to make the numbers closer to 1.

@njit(cache=True)
def rad_a(i, n_inv, D):
    n = i + n_inv - 1
    return -(n*n + (D[0] + 1.)*n + D[0])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

@njit(cache=True)
def rad_b(i, n_inv, D):
    if (i==0): return 0
    n = i + n_inv
    return (-2.*n*n + (D[1] + 2.)*n + D[3])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

#Note that we do not jit the following function, since lentz is not jitted.

def leaver_cf_inv_lentz_old(omega, a, s, m, A, n_inv,
                            tol=1.e-10, N_min=0, N_max=np.Inf):
    """ Legacy function. Same as :meth:`leaver_cf_inv_lentz` except
    calling :meth:`qnm.contfrac.lentz`.  We do not jit this function
    since lentz is not jitted.  It remains here for testing purposes.
    See documentation for :meth:`leaver_cf_inv_lentz` for parameters
    and return value.

    Examples
    --------

    >>> from qnm.radial import leaver_cf_inv_lentz_old, leaver_cf_inv_lentz
    >>> print(leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0))
    ((-3.5662773770495972-1.538871079338485j), 9.702532283649582e-11, 76)

    Compare the two versions of the function:

    >>> old = leaver_cf_inv_lentz_old(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
    >>> new = leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0)
    >>> [ old[i]-new[i] for i in range(3)]
    [0j, 0.0, 0]

    """

    D = D_coeffs(omega, a, s, m, A)

    # This is only use for the terminating fraction
    n = np.arange(0, n_inv+1)
    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    conv2, cf_err, n_frac = lentz(rad_a, rad_b,
                                  args=(n_inv, D),
                                  tol=tol, N_min=N_min, N_max=N_max)

    return (beta[n_inv]
            - gamma[n_inv] * conv1
            + gamma[n_inv] * conv2), cf_err, n_frac


@njit(cache=True)
def leaver_cf_inv_lentz(omega, a, s, m, A, n_inv,
                        tol=1.e-10, N_min=0, N_max=np.Inf):
    """Compute the n_inv inversion of the infinite continued
    fraction for solving the radial Teukolsky equation, using
    modified Lentz's method.
    The value returned is Eq. (44) of [1]_.

    Same as :meth:`leaver_cf_inv_lentz_old`, but with Lentz's method
    inlined so that numba can speed things up.

    Parameters
    ----------
    omega: complex
      The complex frequency for evaluating the infinite continued
      fraction.

    a: float
      Spin parameter of the black hole, 0. <= a < 1 .

    s: int
      Spin weight of the field (i.e. -2 for gravitational).

    m: int
      Azimuthal number for the perturbation.

    A: complex
      Separation constant between angular and radial ODEs.

    n_inv: int
      Inversion number for the infinite continued fraction. Finding
      the nth overtone is typically most stable when n_inv = n .

    tol: float, optional [default: 1.e-10]
      Tolerance for termination of Lentz's method.

    N_min: int, optional [default: 0]
      Minimum number of iterations through Lentz's method.

    N_max: int or comparable, optional [default: np.Inf]
      Maximum number of iterations for Lentz's method.

    Returns
    -------
    (complex, float, int)
      The first value (complex) is the nth inversion of the infinite
      continued fraction evaluated with these arguments. The second
      value (float) is the estimated error from Lentz's method. The
      third value (int) is the number of iterations of Lentz's method.

    Examples
    --------

    >>> from qnm.radial import leaver_cf_inv_lentz
    >>> print(leaver_cf_inv_lentz(omega=.4 - 0.2j, a=0.02, s=-2, m=2, A=4.+0.j, n_inv=0))
    ((-3.5662773770495972-1.538871079338485j), 9.702532283649582e-11, 76)

    References
    ----------
    .. [1] GB Cook, M Zalutskiy, "Gravitational perturbations of the
       Kerr geometry: High-accuracy study," Phys. Rev. D 90, 124021
       (2014), https://arxiv.org/abs/1410.7698 .

    """

    D = D_coeffs(omega, a, s, m, A)

    # This is only use for the terminating fraction
    n = np.arange(0, n_inv+1)
    alpha =     n*n + (D[0] + 1.)*n + D[0]
    beta  = -2.*n*n + (D[1] + 2.)*n + D[3]
    gamma =     n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.

    conv1 = 0.
    for i in range(0, n_inv): # n_inv is not included
        conv1 = alpha[i] / (beta[i] - gamma[i] * conv1)

    ##############################
    # Beginning of Lentz's method, inlined

    # TODO should tiny be a parameter?
    tiny = 1.e-30

    # This is starting with b_0 = 0 for the infinite continued
    # fraction. I could have started with other values (e.g. b_i
    # evaluated with i=0) but then I would have had to subtract that
    # same quantity away from the final result. I don't know if this
    # affects convergence.
    f_old = tiny

    C_old = f_old
    D_old = 0.

    conv = False

    j = 1
    n = n_inv

    while ((not conv) and (j < N_max)):

        # In defining the below a, b sequences, I have cleared a fraction
        # compared to the usual way of writing the radial infinite
        # continued fraction. The point of doing this was that so both
        # terms, a(n) and b(n), tend to 1 as n goes to infinity. Further,
        # We can analytically divide through by n in the numerator and
        # denominator to make the numbers closer to 1.
        an = -(n*n + (D[0] + 1.)*n + D[0])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)
        n = n + 1
        bn = (-2.*n*n + (D[1] + 2.)*n + D[3])/(n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.)

        D_new = bn + an * D_old

        if (D_new == 0):
            D_new = tiny

        C_new = bn + an / C_old

        if (C_new == 0):
            C_new = tiny

        D_new = 1./D_new
        Delta = C_new * D_new
        f_new = f_old * Delta

        if ((j > N_min) and (np.abs(Delta - 1.) < tol)): # converged
            conv = True

        # Set up for next iter
        j = j + 1
        D_old = D_new
        C_old = C_new
        f_old = f_new

    conv2 = f_new

    ##############################

    return (beta[n_inv]
            - gamma[n_inv] * conv1
            + gamma[n_inv] * conv2), np.abs(Delta-1.), j-1































def indexed_alpha(omega, a, s, m, A, n):

    D = D_coeffs(omega, a, s, m, A)

    return n*n + (D[0] + 1.)*n + D[0]


def indexed_beta(omega, a, s, m, A, n):

    D = D_coeffs(omega, a, s, m, A)

    return -2.*n*n + (D[1] + 2.)*n + D[3]


def indexed_gamma(omega, a, s, m, A, n):

    D = D_coeffs(omega, a, s, m, A)

    return n*n + (D[2] - 3.)*n + D[4] - D[2] + 2.


def indexed_a(n, omega, a, s, m, A):

    return -indexed_alpha(omega, a, s, m, A, n-1)*indexed_gamma(omega, a, s, m, A, n)

def indexed_b(n, omega, a, s, m, A):

    return indexed_beta(omega, a, s, m, A, n)


################################################################
################################################################

def dD_da(omega, a, s, m, A):

    M = 1.
    dD_array = [0.j] * 5
    i = complex(0,1.)
    a2 = a*a
    M2 = M*M

    v1 = M2 - a2
    v1_sqrt = np.power(v1, 1./2.)
    v2 = (a-M)*(a+M)
    v1_radical = np.power(v1, 3./2.)



    dD_array[0] =  i*M2*(m - 2.*a*omega)/(v1_radical)
    dD_array[1] = - 2*i*(m*M2 - 2.*a2*a*omega )/v1_radical
    dD_array[2] =  dD_array[0]


    dD_array[3] = (-i*m*M2 + 2.*(i*a2*a - 2.*m*M2*M + m*v2*v1_sqrt )*omega + \
        2.*a*(4.*a2*M + v2*v1_sqrt)*omega*omega)/v1_radical


    dD_array[4] =  M2*(m-2.*a*omega)*(i+4.*M*omega)/v1_radical


    return dD_array

def dD_domega(omega, a, s, m, A):

    M = 1.

    dD_array = [0.j] * 5
    i = complex(0,1.)
    a2 = a*a
    M2 = M*M

    v1 = M2 - a2
    v1_radical = np.power(v1, 3./2.)
    v1_sqrt = np.power(v1, 1./2.)
    v2 = v1_sqrt + M



    dD_array[0] =  -2.*i*M*v2/v1_sqrt
    dD_array[1] =   4.*i*v2*v2/v1_sqrt
    dD_array[2] =  - (2.*i*M*(M + 3.*(-M + v2)))/v1_sqrt

    dD_array[3] = - 2.*(a*m*(M+v2) - 2.*M*v2*(i+8.*M*omega) + \
        a2*(i + 7.*M*omega + v2*omega))/v1_sqrt

    dD_array[4] = -2.*M*(-2.*a*m + i*(3.+2.*s)*v1_sqrt + 8.*M2*omega + M*(i + 8.*v1_sqrt*omega))/v1_sqrt

    return dD_array


def dD_dA(omega, a, s, m, A):

    dD_array = [0.j] * 5

    dD_array[0] =  0.
    dD_array[1] =  0.
    dD_array[2] =  0.
    dD_array[3] =  -1.
    dD_array[4] =  0.

    return dD_array


def dalpha_da(omega, a, s, m, A, n):
    return dD_da(omega, a, s, m, A)[0]*(n+1.)

def dalpha_domega(omega, a, s, m, A, n):
    return dD_domega(omega, a, s, m, A)[0]*(n+1.)

def dalpha_dA(omega, a, s, m, A, n):
    return dD_dA(omega, a, s, m, A)[0]*(n+1.)

def dbeta_da(omega, a, s, m, A, n):
    return dD_da(omega, a, s, m, A)[1]*n + dD_da(omega, a, s, m, A)[3]

def dbeta_domega(omega, a, s, m, A, n):
    return dD_domega(omega, a, s, m, A)[1]*n + dD_domega(omega, a, s, m, A)[3]

def dbeta_dA(omega, a, s, m, A, n):
    return dD_dA(omega, a, s, m, A)[1]*n + dD_dA(omega, a, s, m, A)[3]

def dgamma_da(omega, a, s, m, A, n):
    return dD_da(omega, a, s, m, A)[2]*(n-1) + dD_da(omega, a, s, m, A)[4]

def dgamma_domega(omega, a, s, m, A, n):
    return dD_domega(omega, a, s, m, A)[2]*(n-1) + dD_domega(omega, a, s, m, A)[4]

def dgamma_dA(omega, a, s, m, A, n):
    return dD_dA(omega, a, s, m, A)[2]*(n-1) + dD_dA(omega, a, s, m, A)[4]


def da_da(n, omega, a, s, m, A):
    return -(indexed_gamma(omega, a, s, m, A, n)*dalpha_da(omega, a, s, m, A, n-1) \
        + indexed_alpha(omega, a, s, m, A, n-1)*dgamma_da(omega, a, s, m, A, n) )

def da_domega(n, omega, a, s, m, A):
    return -(indexed_gamma(omega, a, s, m, A, n)*dalpha_domega(omega, a, s, m, A, n-1) \
        + indexed_alpha(omega, a, s, m, A, n-1)*dgamma_domega(omega, a, s, m, A, n))

def da_dA(n, omega, a, s, m, A):
    return -(indexed_gamma(omega, a, s, m, A, n)*dalpha_dA(omega, a, s, m, A, n-1) \
        + indexed_alpha(omega, a, s, m, A, n-1)*dgamma_dA(omega, a, s, m, A, n))


def db_da(n, omega, a, s, m, A):
    return dbeta_da(omega, a, s, m, A, n)

def db_domega(n, omega, a, s, m, A):
    return dbeta_domega(omega, a, s, m, A, n)

def db_dA(n, omega, a, s, m, A):
    return dbeta_dA(omega, a, s, m, A, n)


def lentz_with_grad(a, b, da, db,
                    args=(),
                    tol=1.e-10,
                    N_min=0, N_max=np.Inf,
                    tiny=1.e-30):
    """Compute a continued fraction (and its derivative) via modified
    Lentz's method.

    This implementation is by the book [1]_.  The value to compute is:
      b_0 + a_1/( b_1 + a_2/( b_2 + a_3/( b_3 + ...)))
    where a_n = a(n, *args) and b_n = b(n, *args).

    Parameters
    ----------
    a: callable returning numeric.
    b: callable returning numeric.
    da: callable returning array-like.
    db: callable returning array-like.

    args: tuple [default: ()]
      Additional arguments to pass to the user-defined functions a, b,
      da, and db.  If given, the additional arguments are passed to
      all user-defined functions, e.g. `a(n, *args)`.  So if, for
      example, `a` has the signature `a(n, x, y)`, then `b` must have
      the same  signature, and `args` must be a tuple of length 2,
      `args=(x,y)`.

    tol: float [default: 1.e-10]
      Tolerance for termination of evaluation.

    N_min: int [default: 0]
      Minimum number of iterations to evaluate.

    N_max: int or comparable [default: np.Inf]
      Maximum number of iterations to evaluate.

    tiny: float [default: 1.e-30]
      Very small number to control convergence of Lentz's method when
      there is cancellation in a denominator.

    Returns
    -------
    (float, array-like, float, int)
      The first element of the tuple is the value of the continued
      fraction.
      The second element is the gradient.
      The third element is the estimated error.
      The fourth element is the number of iterations.

    References
    ----------
    .. [1] WH Press, SA Teukolsky, WT Vetterling, BP Flannery,
       "Numerical Recipes," 3rd Ed., Cambridge University Press 2007,
       ISBN 0521880688, 9780521880688 .

    """

    if not isinstance(args, tuple):
        args = (args,)

    f_old = b(0, *args)

    if (f_old == 0):
        f_old = tiny

    C_old = f_old
    D_old = 0.

    # f_0 = b_0, so df_0 = db_0
    df_old = db(0, *args)
    dC_old = df_old
    dD_old = 0.

    conv = False

    j = 1

    while ((not conv) and (j < N_max)):

        aj, bj = a(j, *args), b(j, *args)
        daj, dbj = da(j, *args), db(j, *args)

        # First: modified Lentz
        D_new = bj + aj * D_old

        if (D_new == 0):
            D_new = tiny
        D_new = 1./D_new

        C_new = bj + aj / C_old

        if (C_new == 0):
            C_new = tiny

        Delta = C_new * D_new
        f_new = f_old * Delta

        # Second: the derivative calculations
        # The only possibly dangerous denominator is C_old,
        # but it can't be 0 (at worst it's "tiny")
        dC_new = dbj + (daj*C_old - aj*dC_old)/(C_old*C_old)
        dD_new = -D_new*D_new*(dbj + daj*D_old + aj*dD_old)
        df_new = df_old*Delta + f_old*dC_new*D_new + f_old*C_new*dD_new

        # Did we converge?
        if ((j > N_min) and (np.abs(Delta - 1.) < tol)):
            conv = True

        # Set up for next iter
        j      = j + 1
        C_old  = C_new
        D_old  = D_new
        f_old  = f_new
        dC_old = dC_new
        dD_old = dD_new
        df_old = df_new

    # Success or failure can be assessed by the user
    return f_new, df_new, np.abs(Delta - 1.), j-1
















##########
