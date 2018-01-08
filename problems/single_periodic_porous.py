import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Charged
from porous import Obstacles
from mpi4py import MPI
__author__ = "Gaute Linga"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class PeriodicBoundary(df.SubDomain):
    # Left boundary is target domain
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool((df.near(x[0], -self.Lx/2) or
                     df.near(x[1], -self.Ly/2)) and
                    (not ((df.near(x[0], -self.Lx/2) and
                           df.near(x[1], self.Ly/2)) or
                          (df.near(x[0], self.Lx/2) and
                           df.near(x[1], -self.Ly/2))))
                    and on_boundary)

    def map(self, x, y):
        if df.near(x[0], self.Lx/2) and df.near(x[1], self.Ly/2):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif df.near(x[0], self.Lx/2):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - self.Ly


def problem():
    info_cyan("Fully periodic porous media flow with electrohydrodynamics.")

    solutes = [["c_p",  1, 0.01, 0.01, 0., 0.],
               ["c_m", -1, 0.01, 0.01, 0., 0.]]

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_single_porous",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.05,
        t_0=0.,
        T=10.0,
        N=32,
        solutes=solutes,
        Lx=3.,  # 8.,
        Ly=3.,  # 8.,
        rad=0.2,
        num_obstacles=16,  # 100,
        grid_spacing=0.05,
        #
        density=[1., 1.],
        viscosity=[0.05, 0.05],
        permittivity=[2., 2.],
        surface_charge=1.0,
        composition=[0.45, 0.55],  # must sum to one
        #
        EC_scheme="NL2",
        use_iterative_solvers=True,
        V_lagrange=True,
        p_lagrange=False,
        c_lagrange=True,
        #
        grav_const=0.2,
        grav_dir=[1., 0.],
        c_cutoff=0.1
    )
    return parameters


def constrained_domain(Lx, Ly, **namespace):
    return PeriodicBoundary(Lx, Ly)


def mesh(Lx=8., Ly=8., rad=0.25, num_obstacles=100,
         grid_spacing=0.05, **namespace):
    mesh = load_mesh(
        "meshes/periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}.h5".format(
            Lx, Ly, rad, num_obstacles, grid_spacing))
    return mesh


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               num_obstacles, rad,
               surface_charge,
               composition,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    # Enforcing the compatibility condition.
    total_charge = num_obstacles*2*np.pi*rad*surface_charge
    total_area = Lx*Ly - num_obstacles*np.pi*rad**2

    sum_zx = sum([composition[i]*solutes[i][1]
                  for i in range(len(composition))])
    C = -(total_charge/total_area)/sum_zx

    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for i, solute in enumerate(solutes):
                c_init_expr = df.Expression(
                    "c0",
                    c0=composition[i]*C,
                    degree=2)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
    return w_init_field


def create_bcs(Lx, Ly, mesh, grid_spacing, rad, num_obstacles,
               surface_charge, solutes, enable_NS, enable_EC,
               p_lagrange, V_lagrange,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    data = np.loadtxt(
        "meshes/periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}.dat".format(
            Lx, Ly, rad, num_obstacles, grid_spacing))
    centroids = data[:, :2]
    rad = data[:, 2]

    # Find a single node to pin pressure to
    x_loc = np.array(mesh.coordinates())
    x_proc = np.zeros((size, 2))
    ids_notboun = np.logical_and(
        x_loc[:, 0] > x_loc[:, 0].min() + df.DOLFIN_EPS,
        x_loc[:, 1] > x_loc[:, 1].min() + df.DOLFIN_EPS)
    x_loc = x_loc[ids_notboun, :]
    d2 = (x_loc[:, 0]+Lx/2)**2 + (x_loc[:, 1]+Ly/2)**2
    x_bottomleft = x_loc[d2 == d2.min()][0]
    x_proc[rank, :] = x_bottomleft
    x_pin = np.zeros_like(x_proc)
    comm.Allreduce(x_proc, x_pin, op=MPI.SUM)
    x_pin = x_pin[x_pin[:, 0] == x_pin[:, 0].min(), :][0]

    info("Pinning point: {}".format(x_pin))

    pin_code = ("x[0] < {x}+{eps} && "
                "x[0] > {x}-{eps} && "
                "x[1] > {y}-{eps} && "
                "x[1] < {y}+{eps}").format(
                    x=x_pin[0], y=x_pin[1], eps=1e-3)

    boundaries = dict(
        obstacles=[Obstacles(Lx, centroids, rad, grid_spacing)]
    )

    # Allocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    noslip = Fixed((0., 0.))

    if enable_NS:
        bcs["obstacles"]["u"] = noslip

        if not p_lagrange:
            bcs_pointwise["p"] = (0., pin_code)

    if enable_EC:
        bcs["obstacles"]["V"] = Charged(surface_charge)

        if not V_lagrange:
            bcs_pointwise["V"] = (0., pin_code)

    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def integrate_bulk_charge(x_, solutes, dx):
    total_bulk_charge = []
    for solute in solutes:
        total_bulk_charge.append(df.assemble(solute[1]*x_[solute[0]]*dx))
    return sum(total_bulk_charge)


def start_hook(w_, x_,
               newfolder, field_to_subspace, field_to_subproblem,
               boundaries,
               boundary_to_mark,
               dx, ds,
               surface_charge, solutes,
               **namespace):
    total_surface_charge = df.assemble(
        df.Constant(surface_charge)*ds(
            boundary_to_mark["obstacles"]))
    info("Total surface charge: {}".format(total_surface_charge))
    total_bulk_charge = integrate_bulk_charge(x_, solutes, dx)
    info("Total bulk charge:    {}".format(total_bulk_charge))
    rescale_factor = -total_surface_charge/total_bulk_charge
    info("Rescale factor:       {}".format(rescale_factor))

    subproblem = field_to_subproblem[solutes[0][0]][0]
    w_[subproblem].vector()[:] = rescale_factor*w_[subproblem].vector().array()

    total_bulk_charge_after = integrate_bulk_charge(x_, solutes, dx)
    info("Final bulk charge:    {}".format(total_bulk_charge_after))

    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
