import meshio
import numpy as np

msh = meshio.read("Spiral3D02.msh")

print(msh.cells)


meshio.write("mesh_Spiral3D.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
meshio.write("mf_Spiral3D.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells["triangle"]}, cell_data={"triangle": {"name_to_read": msh.cell_data["triangle"]["gmsh:physical"]}}))

from dolfin import *
mesh = Mesh()
with XDMFFile("mesh_Spiral3D.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("mf_Spiral3D.xdmf") as infile: infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# Insert physical label number to test subdomain retrival
It_facet = SubsetIterator(mf,14)
