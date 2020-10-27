import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root, load_mesh, mpi_barrier, mpi_comm, mpi_bcast, mpi_gather
from common.bcs import Fixed, Pressure, NoSlip, ContactAngle
#
from ufl import max_value
__author__ = "Matthew Hockley"

####################
#
# This is a 2D demo for simulating micro-droplet formation from a T-Junction.
# The demo uses an imported mesh created in GMSH and converted with Meshio.
#
# The parameter space was adapted from https://youtu.be/HtwWseX-zVM
# NOTE: The geomerties are slightly different so it is not an exact 1:1 copy
#
# Surface Tension = https://youtu.be/HtwWseX-zVM?t=267
# Contact Angle = https://youtu.be/HtwWseX-zVM?t=434
# Oil Velocity = https://youtu.be/HtwWseX-zVM?t=397
# Oil Density and Viscosity = https://youtu.be/HtwWseX-zVM?t=341
# Water Velocity = https://youtu.be/HtwWseX-zVM?t=410
# Water Density and Viscosity = https://youtu.be/HtwWseX-zVM?t=321
#
# Matthew Hockley, 2020, University of Kent, SPS
####################

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
    info_cyan("Flow Focusing T-Junction.")
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 1e-2, 4., 1.],
               ["c_m", -1, 1e-4, 1e-2, 4., 1.]]
    #solutes = [["c_p",  0, 1e-3, 1e-2, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 2 # Water to oil ratio (1:Factor)
    scaling_factor = 0.001 # convert from mm to metres
    
    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic", # Type of problem sovler
        folder="results_flow_focusing", # Save folder
        import_mesh = True, # If importing XDMF mesh files
        scale_factor = scaling_factor, # Change mesh dimension (Use if mesh not in metres)
        mesh_file = "meshes/mesh_flowfocus.xdmf", # Mesh filepath
        subdomains_file = "meshes/mf_flowfocus.xdmf", # Subdomains filepath
        name_Facet = "inlet", # Name of inlet within "boundaries_Facet" for Hmin/H
        restart_folder=False, # Use if restarting from different folder
        enable_NS=True, # Enable Navier Stokes (NS)
        enable_PF=True, # Enable Phase Field (PF)
        enable_EC=False, # Enable Electrochem (EC)
        save_intv=5, # Export data time point interval
        stats_intv=5, # Export stats interval
        checkpoint_intv=50, # Export checkpoint for restart
        tstep=0, # Unsure
        dt=1e-5, # s Time steps
        t_0=0., # s Start time
        T=8., # s Total time
        interface_thickness=(0.07/5)*scaling_factor, # Interface thickness between PF
        solutes=solutes, # I believe are electrochem (EC)related
        base_elements=base_elements, # Basic "CG"/"Lagrange" function space
        WaterOilInlet=0, # 0 = all inlets, 1 = WaterInlet, 2 = OilInlet
        # Assumption that both OilInlets are mirrored in Y axis
        H=[0.41,0.41], # Length of inlet (Updated in "FaceLength()")
        Hmin = [0,0], # Minimum of inlet (Updated in "FaceLength()")
        dim = 2, # Dimensions
        XInflow = True, # Direction of flow along X axis
        concentration_left=1., # Concentration in PF (EC related)
        #
        # Contact Angle required normalisation against 180 deg
        #   as it was defined for phase 2 to phase 1.
        # Here it is phase 1 to phase 2
        contact_angle=180-135, # Deg
        surface_tension=0.005,  # n/m
        contact_angle_init=False, # Not requried
        grav_const=0.0, # N/A for microfluidics
        inlet_velocity=0.0103, #m/s
        inlet_velocityOil=0.0103, #m/s
        V_0=0.,
        #
        # PF1 = Oil (1 in PF data); PF2 = Water (-1 in PF data);
        pf_mobility_coeff=2e-6*scaling_factor, # Important for forming phase liquids
        density=[1000, 998.2], # Kg/m3
        viscosity=[6.71e-3,1.003e-3],# Kg/m.s 
        permittivity=[1., 1.], # EC?
        #
        use_iterative_solvers=True, # if False, might have memory issues
        use_pressure_stabilization=False, # Seems to be a type of SUPG, unsure (see solver)
        #
        # Boundary related physical labels (Numbers within mf_subdomains.xdmf)
        #   Typically created within GMSH/Netgen and converted by Meshio
        boundaries_Facet = {'inlet': 12,
                            'inletT': 9,
                            'inletB': 15,
                            'outlet': 4,
                            'wallLR' : [13,11,7,1,3,5],
                            'wallRL' : [14,10,8,16,2,6]
                            }
    )

    # Retrieve inlet dimensions (min/length) from mesh
    [mesh1, parameters1] = mesh(parameters["mesh_file"], 
        parameters["subdomains_file"], parameters["XInflow"],
        parameters["boundaries_Facet"], "inlet", parameters["scale_factor"], False)
    
    # Remove temp mesh, not required
    mesh1 = []

    # Save parameters to main dictionary (Already bcast from mesh function)
    parameters["dim"] = parameters1["dim"]
    parameters["H"][0] = parameters1["H"]
    parameters["Hmin"][0] = parameters1["Hmin"]

    # In this case, inletT and inletB are indistinguishable with regards to y axis
    #   XInflow = False
    #   Boundary name = "inletT"
    [mesh1, parameters1] = mesh(parameters["mesh_file"], 
        parameters["subdomains_file"], False,
        parameters["boundaries_Facet"], "inletT", parameters["scale_factor"], False)
    
    mesh1 = []
    parameters["H"][1] = parameters1["H"]
    parameters["Hmin"][1] = parameters1["Hmin"]

    # Ensure all processes complete before return (Might be redundant)
    mpi_barrier()

    return parameters


def mesh(mesh_file, subdomains_file, XInflow,
            boundaries_Facet, name_Facet, scale_factor,
            import_mesh, **namespace):
    # Load mesh from file (NETGEN mesh as .grid to .xml using DOLFIN)
    
    mesh = df.Mesh()
    with df.XDMFFile(mesh_file) as infile:
        infile.read(mesh)

    # # Scale mesh from mm to m
    x = mesh.coordinates()
    #scaling_factor = 0.001
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
    mesh.bounding_box_tree().build(mesh) # development version

    # Define boundary conditions
    dim = mesh.topology().dim()
    if mpi_is_root():   
        print('Dim:',dim)

    # Ensure all processes have completed
    mpi_barrier()

    if import_mesh: #Mesh import is true
        return mesh
    else: #Otherwise generating range and min of boundary facet assuming line
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
        if XInflow == True:
            parameters["H"] = Y
            parameters["Hmin"] = Ymin
        else:
            parameters["H"] = X
            parameters["Hmin"] = Xmin

        # Ensure all processes have completed (Might be redundant)
        mpi_barrier()

        return mesh, parameters


def initialize(H, Hmin,
               interface_thickness, solutes, restart_folder,
               field_to_subspace, inlet_velocityOil,
               inlet_velocity, concentration_left,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state.velocity
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.
    Note: You only need to specify the initial states that are nonzero.
    """
    w_init_field = dict()
    if not restart_folder:
        if enable_NS:
            try:
                subspace = field_to_subspace["u"].collapse()
            except:
                subspace = field_to_subspace["u"]
            #length inlet, water inflow,
            #   X (0) or Y (1) dir flow, 
            #   Positive/neg flow along axis (+1/-1),
            #   Hmin value
            u_init = velocity_init(H[0], inlet_velocity, 0, 1, Hmin[0])
            w_init_field["u"] = df.interpolate(u_init, subspace)
        # Phase field
        if enable_PF:
            w_init_field["phi"] = df.interpolate(
                df.Constant(1.),
                field_to_subspace["phi"].collapse())

    return w_init_field


def create_bcs(dim, H, Hmin, inlet_velocity, inlet_velocityOil,
               V_0, solutes, subdomains_file, WaterOilInlet,
               concentration_left,
               interface_thickness,
               enable_NS, enable_PF, enable_EC, 
               mesh, boundaries_Facet, contact_angle, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    mvc = df.MeshValueCollection("size_t", mesh, dim-1) 
    with df.XDMFFile(subdomains_file) as infile:
        infile.read(mvc, "name_to_read")
    facet_domains = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Re-create boundaries with facet_domain for mesh relevance

    boundaries = dict(
        inlet = [facet_domains, boundaries_Facet["inlet"]],
        inletT = [facet_domains, boundaries_Facet["inletT"]],
        inletB = [facet_domains, boundaries_Facet["inletB"]],
        outlet = [facet_domains, boundaries_Facet["outlet"]],
        wallLR = [facet_domains, boundaries_Facet["wallLR"]],
        wallRL = [facet_domains, boundaries_Facet["wallRL"]]
    )

     # Alocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    ### Velocity has 3 inlets in this example due
    #       to the flow focusing pinching aspect

    ## Velocity Phase Flow In
    #length inlet, water inflow, X or Y, Positive/neg flow along axis
    if not WaterOilInlet == 2:
        velocity_expr = velocity_init(H[0], inlet_velocity, 0, 1, Hmin[0]) 
        velocity_in = Fixed(velocity_expr)
        if enable_NS:
            bcs["inlet"]["u"] = velocity_in
            if WaterOilInlet == 1:
                bcs["inletT"]["u"] = NoSlip()
                bcs["inletB"]["u"] = NoSlip()

    ## Velocity Top In
    #length inlet, water inflow, X or Y, Positive/neg flow along axis
    if not WaterOilInlet == 1:
        velocity_expr = velocity_init(H[1], inlet_velocityOil, 1, -1, Hmin[1]) 
        velocity_inT = Fixed(velocity_expr)


    ## Velocity Bottom In
    #length inlet, water inflow, X or Y, Positive/neg flow along axis
    if not WaterOilInlet == 1:
        velocity_expr = velocity_init(H[1], inlet_velocityOil, 1, 1, Hmin[1]) 
        velocity_inB = Fixed(velocity_expr)
        if enable_NS:
            bcs["inletT"]["u"] = velocity_inT
            bcs["inletB"]["u"] = velocity_inB
            if WaterOilInlet == 2:
                bcs["inlet"]["u"] = NoSlip()


    pressure_out = Pressure(0.0)
    noslip = NoSlip()

    V_left = Fixed(V_0)
    V_right = Fixed(0.)

    ## Define boundaries
    #    Note we have one outlet and two sets of walls
    #       from experience (FEniCS), opposite boundaries can
    #       behave badly when all grouped   
    if enable_NS:
        bcs["outlet"]["p"] = pressure_out
        bcs["wallLR"]["u"] = noslip
        bcs["wallRL"]["u"] = noslip

    # Phase field uses an expersion `tanH` which defines PF drop off
    if enable_PF:
        phi_expr = df.Expression(
            "tanh((abs((x[1]-Hmin)-H/2)-H/16)/(sqrt(2)*eps))",
            H=H[0], Hmin = Hmin[0], eps=interface_thickness,
            degree=2)
        phi_inlet = Fixed(phi_expr)

        ## PF Fixed across boundary
        # as no water can enter oil inlet
        #   and vice-versa
        bcs["inlet"]["phi"] = Fixed(df.Constant(-1.))
        bcs["inletT"]["phi"] = Fixed(df.Constant(1.))
        bcs["inletB"]["phi"] = Fixed(df.Constant(1.))
        ## Add contact angle to NS No-Slip Boudnaries
        bcs["wallLR"]["phi"] = ContactAngle(contact_angle)
        bcs["wallRL"]["phi"] = ContactAngle(contact_angle)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, rad, eps, function_space):
    # Phase field uses an expersion `tanH` which defines PF drop off
    # Not dependent on `H` and `Hmin` due to evolution over time
    #   in channel
    expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad,
                                  eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def velocity_init(H, inlet_velocity, XY, Pos, xyMin, degree=2):
    #length inlet, water inflow, X or Y, Positive/neg flow along axis
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


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))
    # Function which runs every simulation tick

def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    # func = 1.-phi**2 + 0.0001
    # return 0.75 * gamma * max_value(func, 0.)
    return gamma
    # Function to control PF mobility over time. 


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
    # Function which runs at start of simulation