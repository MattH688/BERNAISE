import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root, load_mesh, mpi_barrier, mpi_comm, mpi_bcast, mpi_gather, get_dt_CFL
# from common.cmd import MPI_rank
# import mpi4py
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
    Z = []
    xInflowRange = 0
    yInflowRange = 0
    zInflowRange = 0
    xInflowMin = 0
    yInflowMin = 0
    zInflowMin = 0

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
            Z.append(v.point().z())

    # Ensure all processes collect co-ords for desired face
    mpi_barrier()

    # Gather all co-ords to calc length/min
    X = mpi_gather(X, 0)
    Y = mpi_gather(Y, 0)
    Z = mpi_gather(Z, 0)

    # Sync all parallel processes for length/min calc
    mpi_barrier()

    if mpi_is_root():
        # Remove empty and combine all arrays
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        # Calculate length and min values
        xInflowRange = np.ptp(X,axis=0)
        yInflowRange = np.ptp(Y,axis=0)
        zInflowRange = np.ptp(Z,axis=0)
        xInflowMin = np.amin(X)
        yInflowMin = np.amin(Y)
        zInflowMin = np.amin(Z)

    # END: Sync all parallel processes for length/min calc
    mpi_barrier()

    # Broadcast all length/min calc to all nodes used
    xInflowRange = mpi_bcast(xInflowRange, 0)
    yInflowRange = mpi_bcast(yInflowRange, 0)
    zInflowRange = mpi_bcast(zInflowRange, 0)
    xInflowMin = mpi_bcast(xInflowMin, 0)
    yInflowMin = mpi_bcast(yInflowMin, 0)
    zInflowMin = mpi_bcast(zInflowMin, 0)

    # Clear variables
    v = []
    It_facet = []
    facet_domains = []

    return xInflowRange, yInflowRange, zInflowRange, xInflowMin, yInflowMin, zInflowMin

def problem():
    info_cyan("Secondary flow / Dean Drag Forces benchmark.")
    #         2, beta in phase 1, beta in phase 2
    #solutes = [["c_p",  1, 1e-4, 1e-2, 4., 1.],
    #           ["c_m", -1, 1e-4, 1e-2, 4., 1.]]
    solutes = [["c_p",  1, 2.3e-11, 2.3e-11, 0., 0.]]

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
        solver="basic_IPCS_Adj",#"basic_IPCS_Adj" # Type of problem sovler
        folder="results_curvedChannel", # Save folder
        import_mesh = True, # If importing XDMF mesh files
        scale_factor = scaling_factor,  # Change mesh dimension (Use if mesh not in metres)
        mesh_file = "meshes/mesh_curvedMicrochannel.xdmf", # Mesh filepath
        subdomains_file = "meshes/mf_curvedMicrochannel.xdmf", # Subdomains filepath
        name_Facet = "inlet", # Name of inlet within "boundaries_Facet" for Hmin/H
        restart_folder=False, # Use if restarting from different folder
        enable_NS=True, # Enable Navier Stokes (NS)
        enable_PF=False, # Enable Phase Field (PF)
        enable_EC=False, # Enable Electrochem (EC)
        save_intv=5, # Export data time point interval
        stats_intv=5, # Export stats interval
        checkpoint_intv=50, # Export checkpoint for restart
        tstep=0, # Unsure
        dt=4e-6,#0.000625, # s Time steps
        t_0=0., # s Start time
        T=300,# s Total time or number of steps if CFL
        solutes=solutes, # I believe are electrochem (EC)/phase field (PF) related
        base_elements=base_elements, # Basic "CG"/"Lagrange" function space
        #
        H=0.41, # Length of inlet (Updated in "faceLength")
        HZ=0, # Length of inlet 2D dimension (Updated in "faceLength")
        Hmin=0, # Minimum of inlet (Updated in "faceLength")
        HminZ=0, # Minimum of inlet 2D dimension (Updated in "faceLength")
        Hy=0, # Length of inlet for EC related functions
        Hymin=0, # Length of inlet for EC related functions
        dim = 3, # Dimensions
        XYZ = 1, # If XY(2), XZ(1) or YZ(0) direction of flow
        #
        # Simulation parameters
        grav_const=0.0, # 0 gravity as microfluidic
        grav_dir=[0., 0., 0.], # Gravity direction - Needs to be 3D, default 2D
        inlet_velocity=0.1, # m/s (Negative due to -x inflow direction)
        V_0=0., # Unsure
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
        boundaries_Facet = {'inlet': 2, #8
                            'outletL': 7, #9
                            'wall': [3,4,5,6],
                            }
    )

    # Retrieve inlet dimensions (min/length) from mesh
    [mesh1, parameters1] = mesh(parameters["mesh_file"], 
        parameters["subdomains_file"], parameters["XYZ"],
        parameters["boundaries_Facet"], "inlet", parameters["scale_factor"], False)

    # Use CFL (Courant-Friedrichs-Lewy) to calculate time step
    dt = get_dt_CFL(mesh1, parameters["inlet_velocity"])
    parameters["dt"] = dt
    parameters["T"] = parameters["T"]*parameters["dt"]

    # Remove temp mesh, not required
    mesh1 = []

    # Save parameters to main dictionary (Already bcast from mesh function)

    parameters["dim"] = parameters1["dim"]
    parameters["H"] = parameters1["H"]
    parameters["Hmin"] = parameters1["Hmin"]
    parameters["HZ"] = parameters1["HZ"]
    parameters["HminZ"] = parameters1["HminZ"]
    parameters["Hy"] = parameters1["Hy"]
    parameters["Hymin"] = parameters1["Hymin"]

    # Output of Hmin and H for inlet velocity calculations (see "velocity_init")
    # mpi_barrier()
    # if mpi_is_root():
    #     print("Hmin: ", parameters["Hmin"])
    #     print("HminZ: ", parameters["Hmin"])
    #     print("H: ", parameters["H"])
    #     print("HZ: ", parameters["H"])
        
    # Ensure all processes complete before return (Might be redundant)
    mpi_barrier()

    return parameters


def mesh(mesh_file, subdomains_file, XYZ,
            boundaries_Facet, name_Facet, scale_factor,
            import_mesh, **namespace):
    # Load mesh from file (NETGEN mesh as .grid to .xml using DOLFIN)

    mesh = df.Mesh()
    with df.XDMFFile(mpi_comm(), mesh_file) as infile:
        infile.read(mesh)

    # # Scale mesh from mm to m
    x = mesh.coordinates()
    x[:, :] *= scale_factor
    # # Move mesh so co-ords always positive
    #
    xyzmin = x.min(axis=0)
    mpi_barrier()
    xyzmin = np.min(mpi_gather(xyzmin, 0))
    mpi_barrier()
    xyzmin = mpi_bcast(xyzmin, 0)
    mpi_barrier()

    x[:, :] = x[:, :] - xyzmin
    # Apply to mesh
    mesh.bounding_box_tree().build(mesh)

    # Define boundary conditions
    dim = mesh.topology().dim()
    if mpi_is_root():
        print('Dim:',dim)

    # Ensure all processes have completed
    mpi_barrier()

    if import_mesh: #Mesh import is true
        return mesh
    else: #Otherwise generating range and min of boundary facet assuming line

        # Retrieve length and min of boundary facet (inlet in most cases)
        [X, Y, Z, Xmin, Ymin, Zmin] = FaceLength(boundaries_Facet[name_Facet], mesh,
            subdomains_file, dim)

        # Display boundary dimensions (inlet in most cases)
        mpi_barrier()
        if mpi_is_root():
            info_yellow("Boundary Dimensions")
            print("x: ",X)
            print("y: ",Y)
            print("z: ",Z)
            print("xMin: ",Xmin)
            print("yMin: ",Ymin)
            print("zMin: ",Zmin)
            print("Min Cell Size: ",mesh.hmin())
            print("Max Cell Size: ",mesh.hmin())

        # Save length/min to dictionary
        #   This will not overwrite prior dictionary
        #    as this is in an indepenent function
        parameters = dict()
        parameters["dim"] = dim

        # Depending on flow direction (X/Y/Z),
        #   the remainder axes need min/length
        #   for calculating 3D parabolic inflow
        #
        # Hy/Hymin is for setting the appropriate EC boundaries for init. conc.
        if XYZ == 0:
            parameters["H"] = Y
            parameters["Hmin"] = Ymin
            parameters["Hy"] = X
            parameters["Hymin"] = Xmin
        else:
            parameters["H"] = X
            parameters["Hmin"] = Xmin
            parameters["Hy"] = Y
            parameters["Hymin"] = Ymin

        parameters["HZ"] = Z
        parameters["HminZ"] = Zmin

        if XYZ == 3:
            parameters["HZ"] = Y
            parameters["HminZ"] = Ymin
            parameters["Hy"] = Z
            parameters["Hymin"] = Zmin

        # Ensure all processes have completed (Might be redundant)
        mpi_barrier()

        return mesh, parameters


def initialize(H, Hmin, HZ, HminZ, Hy, Hymin, solutes, restart_folder,
               field_to_subspace, XYZ, concentration_init,
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
    # if not restart_folder:
    #     if enable_NS:
    #         try:
    #             subspace = field_to_subspace["u"].collapse()
    #         except:
    #             subspace = field_to_subspace["u"]
    #         #u_init = velocity_init(H, HZ, inlet_velocity, XYZ, 1, Hmin, HminZ)
    #         u_init = df.Expression(("0.0","0.0","0.0"), degree=3)
    #         w_init_field["u"] = df.interpolate(u_init, subspace)
    #     ### Initialize electrochemistry
    #     # Z axis (doamin to pull X-Y plane into 3D) is ignored
    #     # initial_c(xMin, xOffSet, yMin, yOffSet, c_init, function_space):
    #     if enable_EC:
    #         c_init_expr = initial_c(
    #             Hmin, 3*scale_factor, Hymin, 0.25*scale_factor, concentration_init,
    #             field_to_subspace[solutes[0][0]].collapse())
    #         w_init_field[solutes[0][0]] = df.interpolate(c_init_expr,
    #                 field_to_subspace[solutes[0][0]].collapse())

    # Ensure all processes have completed (Might be redundant)
    mpi_barrier()
    return w_init_field


def create_bcs(dim, H, Hmin, HZ, HminZ, XYZ, inlet_velocity,
               V_0, solutes, subdomains_file, concentration_init,
               enable_NS, enable_PF, enable_EC, 
               mesh, boundaries_Facet, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    mvc = df.MeshValueCollection("size_t", mesh, dim-1) 
    with df.XDMFFile(subdomains_file) as infile:
        infile.read(mvc, "name_to_read")
    facet_domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Re-create boundaries with facet_domain for mesh relevance

    boundaries = dict(
        inlet = [facet_domains, boundaries_Facet["inlet"]],
        outletL = [facet_domains, boundaries_Facet["outletL"]],
        wall = [facet_domains, boundaries_Facet["wall"]],
    )

     # Alocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    ## Velocity Phase Flow In (Retrieve expression)
    #
    #length inlet, water inflow, X/Y/Z, Positive/neg flow along axis
    velocity_expr = velocity_init(H, HZ, inlet_velocity, XYZ, 1, Hmin, HminZ) 
    velocity_in = Fixed(velocity_expr)

    # Pressure set to 0 at outlet
    pressure_out = Pressure(0.0)
    # Create NoSlip function for walls
    noslip = Fixed((0., 0., 0.)) # Unlike 2D "NoSlip()", need 3 dimensions

    ## Define boundaries
    #       Note we have two outlets
    if enable_NS:
        bcs["inlet"]["u"] = velocity_in
        bcs["outletL"]["p"] = pressure_out
        bcs["wall"]["u"] = noslip

    ## Define EC boudnaries
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


def velocity_init(H, HZ, inlet_velocity, XYZ, Pos, Hmin, HminZ, dim=3):
    # length inlet, water inflow, X/Y/Z, Positive/neg flow along axis
    # XYZ = XY(2), XZ(1) or YZ(0) boundaries
    if XYZ == 0: # X axis
        return df.Expression(
            ("((((A*4.0*(x[2] - zMin)*(zRange - (x[2] - zMin))) / pow(zRange,2)) + ((A*4.0*(x[1] - yMin)*(yRange - (x[1] - yMin))) / pow(yRange, 2)))/2)","0.0","0.0"),
            A=df.Constant(inlet_velocity), yRange=df.Constant(H), zRange=df.Constant(HZ), yMin=df.Constant(Hmin), zMin=df.Constant(HminZ), degree=dim)   
    elif XYZ == 1: # Y axis
        return df.Expression(
                ("0.0","((((A*4.0*(x[2] - zMin)*(zRange - (x[2] - zMin))) / pow(zRange,2)) + ((A*4.0*(x[0] - xMin)*(xRange - (x[0] - xMin))) / pow(xRange, 2)))/2)","0.0"),
            A=df.Constant(inlet_velocity), xRange=df.Constant(H), zRange=df.Constant(HZ), xMin=df.Constant(Hmin), zMin=df.Constant(HminZ), degree=dim)   
    else: # if XY == 2: # Z axis
        return df.Expression(
            ("0.0","0.0","((((A*4.0*(x[1] - yMin)*(yRange - (x[1] - yMin))) / pow(yRange,2)) + ((A*4.0*(x[0] - xMin)*(xRange - (x[0] - xMin))) / pow(xRange, 2)))/2)"),
            A=df.Constant(inlet_velocity), xRange=df.Constant(H), yRange=df.Constant(HZ), xMin=df.Constant(Hmin), yMin=df.Constant(HminZ), degree=dim)
    # ## If you want a constant and not parabolic inflow, comment above and use...
    #
    # return df.Expression(("0.0","U","0.0"), U=inlet_velocity, degree=dim)
    # Remember to define X/Y/Z inflow manually if constant (current X)

def initial_c(xMin, xOffSet, yMin, yOffSet, c_init, function_space):
    """ Function describing the initial concentration field. """
    # Example here defined a box near the beginning of the spiral

    c_init_expr = df.Expression(('((x[0] > (xMin-xOffSet) && x[0] < xMin) &&'
        '((x[1] > (yMin-yOffSet)) && (x[1] < (yMin+yOffSet)))) ? A : 0'),
        xMin=df.Constant(xMin), xOffSet=df.Constant(xOffSet),
        yMin=df.Constant(yMin), yOffSet=df.Constant(yOffSet),
        A=df.Constant(c_init), degree=2)

    return c_init_expr

def tstep_hook(t, tstep, trial_functions, mesh, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, enable_NS, **namespace):
    info_blue("Timestep = {}".format(tstep))
    # Function which runs every simulation tick

    # Compute error
    if enable_NS:
        info_yellow('max u: {}'.format(w_["NSu"].vector().max()))
        # info_yellow("Worst possible Courant number = {}".format((tstep*(w_["NSu"].vector().max()))/mesh.hmin()))
        

def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
    # Function which runs at start of simulation