import meshio
import numpy as np

msh = meshio.read("Spiral2D.msh")

print(msh.cell_data["line"]["gmsh:physical"])
print(msh.cells)


meshio.write("mesh_Spiral2D.xdmf", meshio.Mesh(points=msh.points[:,:2], cells={"triangle": msh.cells["triangle"]}))
meshio.write("mf_Spiral2D.xdmf", meshio.Mesh(points=msh.points, cells={"line": msh.cells["line"]}, cell_data={"line": {"name_to_read": msh.cell_data["line"]["gmsh:physical"]}}))

from dolfin import *
mesh = Mesh()
with XDMFFile("mesh_Spiral2D.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("mf_Spiral2D.xdmf") as infile: infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# Insert physical label number to test subdomain retrival
It_facet = SubsetIterator(mf,10)
