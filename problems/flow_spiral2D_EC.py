import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root, load_mesh, mpi_barrier, mpi_comm, mpi_bcast, mpi_gather
# from common.cmd import MPI_rank
# from mpi4py import MPI
from common.bcs import Fixed, Pressure, NoSlip
#
from ufl import max_value
__author__ = "Matthew Hockley"


def FaceLength(faceNum, mesh, subdomains_file, dim):

    # State desired face which measures are taking place upon
    if mpi_is_root():
        print(faceNum)

    # Display how mesh is separated
    # print("Node: ", MPI_rank, "Mesh Cells: ", mesh.cells().size)
    
    # Import subdomains
    mvc = df.MeshValueCollection("size_t", mesh, dim-1)
    with df.XDMFFile(mpi_comm(), subdomains_file) as infile:
        infile.read(mvc, "name_to_read")
    facet_domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    ## Calculate limits so inflow parabolic can work on co-ordinates not at 0

    # Create empty variable space
    X = []
    Y = []
    xInflowRange = 0
    yInflowRange = 0
    xInflowMin = 0
    yInflowMin = 0

    # Retrive all co-ords as element for desired face
    It_facet = df.SubsetIterator(facet_domains,faceNum)
    mpi_barrier()
    # print("Faces: ", df.SubsetIterator(facet_domains,faceNum))
    

    #https://fenicsproject.org/qa/13995/print-coordinate-of-boundary-seperately-i-e-left-boundary/
    #It_mesh = vertices([facet_domains.array() == 26])

    # Collected all co-ords for desired face
    for facet_domains in It_facet:
        for v in df.vertices(facet_domains):
            X.append(v.point().x())
            Y.append(v.point().y())

    # Ensure all processes collect co-ords for desired face
    mpi_barrier()

    # Gather all co-ords to calc length/min
    X = mpi_gather(X, 0)
    Y = mpi_gather(Y, 0)

    # Sync all parallel processes for length/min calc
    mpi_barrier()
    
    if mpi_is_root():
        # Remove empty and combine all arrays
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        # Calculate length and min values
        xInflowRange = np.ptp(X,axis=0)
        yInflowRange = np.ptp(Y,axis=0)
        xInflowMin = np.amin(X)
        yInflowMin = np.amin(Y)

    # END: Sync all parallel processes for length/min calc
    mpi_barrier()

    # Broadcast all length/min calc to all nodes used
    xInflowRange = mpi_bcast(xInflowRange, 0)
    yInflowRange = mpi_bcast(yInflowRange, 0)
    xInflowMin = mpi_bcast(xInflowMin, 0)
    yInflowMin = mpi_bcast(yInflowMin, 0)

    # Clear variables
    v = []
    It_facet = []
    facet_domains = []

    return xInflowRange, yInflowRange, xInflowMin, yInflowMin

def problem():
    info_cyan("Flow around 2D spiral benchmark.")
    #      name, phase presence, diff. in phase 1,
    #            diff. in phase 2, solubility energy in phase 1,
    #               solubility energy in phase 2
    solutes = [["c_p", 1, 1e-6, 1e-6, 0., 0.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 2
    scaling_factor = 0.001
    
    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic", # Type of problem sovler
        folder="results_spiral2D", # Save folder
        import_mesh = True, # If importing XDMF mesh files
        scale_factor = scaling_factor, # Change mesh dimension (Use if mesh not in metres)
        mesh_file = "meshes/mesh_Spiral2D.xdmf", # Mesh filepath
        subdomains_file = "meshes/mf_Spiral2D.xdmf", # Subdomains filepath
        name_Facet = "inlet", # Name of inlet within "boundaries_Facet" for Hmin/H
        restart_folder=False, # Use if restarting from different folder
        enable_NS=True, # Enable Navier Stokes (NS)
        enable_PF=False, # Enable Phase Field (PF)
        enable_EC=True, # Enable Electrochem (EC)
        save_intv=5, # Export data time point interval
        stats_intv=5, # Export stats interval
        checkpoint_intv=50, # Export checkpoint for restart
        tstep=0, # Unsure
        dt=0.0001*1e-3, #e-4 0.0015/factor, # s Time steps
        t_0=0., # s Start time
        T=8., # s Total time
        solutes=solutes, # I believe are electrochem (EC)/phase field (PF) related
        base_elements=base_elements, # Basic "CG"/"Lagrange" function space
        #
        H=0.41, # Length of inlet (Updated in "faceLength")
        Hy=0.41,
        Hmin=0, # Minimum of inlet (Updated in "faceLength")
        Hymin=0,
        dim = 2, # Dimensions
        XInflow = True, # Direction of flow along X axis
        #
        # Simulation parameters
        grav_const=0.0, # 0 gravity as microfluidic
        inlet_velocity=1.5*scaling_factor, # m/s (Negative due to -x inflow direction)
        V_0=0., # Default electric field
        #
        # Fluid parameters (Water at 22C)
        density=[998.2, 998.2], # Kg/m3
        viscosity=[1.003e-3, 1.003e-3], # Kg/m.s kinematic viscosity
        permittivity=[1., 1.], # EC?
        concentration_init = 10.,
        #
        # Solver parameters
        use_iterative_solvers=True, # if False, might have memory issues
        use_pressure_stabilization=False, # Seems to be a type of SUPG, unsure (see solver)
        #
        # Boundary related physical labels (Numbers within mf_subdomains.xdmf)
        #   Typically created within GMSH/Netgen and converted by Meshio
        boundaries_Facet = {'inlet': 10,
                            'outletL': 6,
                            'outletR': 3,
                            'wall': [2,4,5,7,8,9],
                            }
    )

    # Retrieve inlet dimensions (min/length) from mesh
    [mesh1, parameters1] = mesh(parameters["mesh_file"], 
    parameters["subdomains_file"], parameters["XInflow"],
    parameters["boundaries_Facet"], "inlet", parameters["scale_factor"], False)
    
    # Remove temp mesh, not required
    mesh1 = []

    # Save parameters to main dictionary (Already bcast from mesh function)
    #  Differs from spiral2D due to creating an initial concentration prescence
    parameters["dim"] = parameters1["dim"]
    parameters["H"] = parameters1["H"]
    parameters["Hmin"] = parameters1["Hmin"]
    parameters["Hy"] = parameters1["Hy"]
    parameters["Hymin"] = parameters1["Hymin"]

    # Ensure all processes complete before return (Might be redundant)
    mpi_barrier()

    return parameters


def mesh(mesh_file, subdomains_file, XInflow,
            boundaries_Facet, name_Facet, scale_factor,
            import_mesh, **namespace):
    # Load mesh from file (NETGEN mesh as .grid to .xml using DOLFIN)
    
    mesh = df.Mesh()
    with df.XDMFFile(mpi_comm(),mesh_file) as infile:
        infile.read(mesh)

    # # Scale mesh from mm to m
    x = mesh.coordinates()
    x[:, :] *= scale_factor
    # # Move mesh so co-ords always positive
    #
    xymin = x.min(axis=0) 
    mpi_barrier()
    xymin = np.min(mpi_gather(xymin, 0))
    mpi_barrier()
    xymin = mpi_bcast(xymin, 0)
    mpi_barrier()

    x[:, :] = x[:, :] - xymin
    # Apply to mesh
    mesh.bounding_box_tree().build(mesh) 

    # Define boundary conditions
    dim = mesh.topology().dim()
    if mpi_is_root():   
        print('Dim:',dim)

    # Ensure all processes have completed
    mpi_barrier()

    if import_mesh: #Mesh import only if true
        return mesh
    else: #Otherwise generating length and min of boundary facet assuming line

        # Retrieve length and min of boundary facet (inlet in most cases)
        [X, Y, Xmin, Ymin] = FaceLength(boundaries_Facet[name_Facet], mesh,
            subdomains_file, dim)

        # Display boundary dimensions (inlet in most cases)
        mpi_barrier()
        if mpi_is_root():
            info_yellow("Boundary Dimensions")
            print("x: ",X)
            print("y: ",Y)
            print("xMin: ",Xmin)
            print("yMin: ",Ymin)

        # Save length/min to dictionary
        #   This will not overwrite prior dictionary
        #    as this is in an indepenent function
        parameters = dict()
        parameters["dim"] = dim
        parameters["Hy"] = Y
        parameters["Hymin"] = Ymin
        parameters["H"] = X
        parameters["Hmin"] = Xmin

        # Ensure all processes have completed (Might be redundant)
        mpi_barrier()

        return mesh, parameters


def initialize(H, Hmin, Hy, Hymin, solutes, restart_folder,
               field_to_subspace, concentration_init,
               inlet_velocity, scale_factor,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state.
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.
    Note: You only need to specify the initial states that are nonzero.
    """
    w_init_field = dict()
    if not restart_folder:
    #     if enable_NS:
    #         try:
    #             subspace = field_to_subspace["u"].collapse()
    #         except:
    #             subspace = field_to_subspace["u"]
    #         u_init = velocity_init(H, inlet_velocity, 0, 1, Hmin)
    #         w_init_field["u"] = df.interpolate(u_init, subspace)

    # Ensure all processes have completed (Might be redundant)

        ### Initialize electrochemistry
        # initial_c(xMin, xOffSet, yMin, yOffSet, c_init, function_space):
        if enable_EC:
            c_init_expr = initial_c(
                Hmin, 3*scale_factor, Hymin, 0.25*scale_factor, concentration_init,
                field_to_subspace[solutes[0][0]].collapse())
            w_init_field[solutes[0][0]] = df.interpolate(c_init_expr,
                 field_to_subspace[solutes[0][0]].collapse())
    mpi_barrier()
    return w_init_field


def create_bcs(dim, H, Hmin, inlet_velocity,
               V_0, solutes, subdomains_file,
               enable_NS, enable_PF, enable_EC, 
               mesh, boundaries_Facet,
               concentration_init, scale_factor,
               Hymin, field_to_subspace, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    mvc = df.MeshValueCollection("size_t", mesh, dim-1) 
    with df.XDMFFile(subdomains_file) as infile:
        infile.read(mvc, "name_to_read")
    facet_domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Re-create boundaries with facet_domain for mesh relevance

    boundaries = dict(
        inlet = [facet_domains, boundaries_Facet["inlet"]],
        outletL = [facet_domains, boundaries_Facet["outletL"]],
        outletR = [facet_domains, boundaries_Facet["outletR"]],
        wall = [facet_domains, boundaries_Facet["wall"]],
    )

     # Alocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    ## Velocity Phase Flow In (Retrieve expression)
    #
    #length inlet, water inflow, X/Y, Positive/neg flow along axis
    velocity_expr = velocity_init(H, inlet_velocity, 0, 1, Hmin) 
    velocity_in = Fixed(velocity_expr)

    # Pressure set to 0 at outlet
    pressure_out = Pressure(0.0)
    # Create NoSlip function for walls
    noslip = NoSlip() # Fixed((0., 0.)) no difference using either.

    ## Define boundaries
    #       Note we have two outlets
    if enable_NS:
        bcs["inlet"]["u"] = velocity_in # Velocity expression for inflow
        bcs["outletL"]["p"] = pressure_out # 0 pressure expression for outflow
        bcs["outletR"]["p"] = pressure_out # 0 pressure expression for outflow
        bcs["wall"]["u"] = noslip # No slip for walls

    #Unused but could make constant concentration for box defined in initialize()
    c_init_expr = initial_c(Hmin, 3*scale_factor,
        Hymin, 0.25*scale_factor, concentration_init,
        field_to_subspace[solutes[0][0]].collapse())

    # Define EC boudnaries
    if enable_EC:
        ## Concentration - Constant at inlet
        bcs["inlet"][solutes[0][0]] = Fixed(concentration_init) 
        ## Parameters "V" for electric field (if defined) 
        # bcs["inlet"]["V"] = Fixed(0.0) V = is electric field
        # bcs["outletL"]["V"] = Fixed(0.0) # No concentration
        # bcs["outletR"]["V"] = Fixed(0.0) # No concentration
        # bcs["wall"]["V"] = Fixed(0.0) # No concentration


    # Ensure all processes have completed (Might be redundant) 
    mpi_barrier()
    return boundaries, bcs, bcs_pointwise


def velocity_init(H, inlet_velocity, XY, Pos, xyMin, degree=2):
    #length inlet, inflow (m/s), X or Y dir, Positive/neg flow along axis
    if XY == 0:
        return df.Expression(
            ("Pos*4*U*(x[1] - xyMin)*(H-(x[1] - xyMin))/pow(H, 2)", "0.0"),
            Pos=Pos, H=H, U=inlet_velocity, xyMin = xyMin, degree=degree)
    else: # if XY == 1
        return df.Expression(
            ("0.0", "Pos*4*U*(x[0] - xyMin)*(H-(x[0] - xyMin))/pow(H, 2)"),
            Pos=Pos, H=H, U=inlet_velocity, xyMin = xyMin, degree=degree)

    ## If you want a constant and not parabolic inflow, comment above and use...
    #
    # return df.Expression(("U","0.0"), U=inlet_velocity, degree=degree)
    # Remember to define X or Y inflow manually if constant (current X)


def initial_c(xMin, xOffSet, yMin, yOffSet, c_init, function_space):
    """ Function describing the initial concentration field. """
    # Example here defined a box near the beginning of the spiral

    c_init_expr = df.Expression(('((x[0] > (xMin-xOffSet) && x[0] < xMin) &&'
        '((x[1] > (yMin-yOffSet)) && (x[1] < (yMin+yOffSet)))) ? A : 0'),
        xMin=df.Constant(xMin), xOffSet=df.Constant(xOffSet),
        yMin=df.Constant(yMin), yOffSet=df.Constant(yOffSet),
        A=df.Constant(c_init), degree=2)

    return c_init_expr


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))
    # Function which runs every simulation tick


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
    # Function which runs at start of simulation