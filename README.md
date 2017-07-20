# BERNAISE
[![Build Status](https://travis-ci.org/gautelinga/BERNAISE.svg?branch=master)](https://travis-ci.org/gautelinga/BERNAISE)
_BERNAISE_ (Binary ElectRohydrodyNAmIc SolvEr) is a flexible, high-level solver of electrohydrodynamic flows in complex geometries currently under development.
It is written in Python and built on the FEniCS project, which in turn effectively interfaces to optimized linear algebra backends such as PETSc.

<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/droplet.gif" width=122 height=254 alt="Buoyancy-driven droplet"/>
    <br /><b>Buoyancy-driven droplet.</b>
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/charged_droplets.gif" width=264 height=87 alt="Colliding oppositely charged droplets"/><br />
    <b>Two colliding oppositely charged droplets.</b> Red: positive charge, blue: negative charge.
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/dielectric_faster.gif" width=192 height=192 alt="Two-phase dielectricum."/><br />
    <b>Two-phase dielectricum/capacitor.</b> Red: positive charge, blue: negative charge. Top: negative surface charge, bottom: positive surface charge.
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/snoevsen.gif" width=250 height=140 alt="Snøvsen."/><br />
    <b>Enhanced oil recovery</b> by application of a surface charge to the pore wall, and an ion concentration in the water phase. The color indicates the charge. The flow is driven by a constant velocity at the top (Couette flow). Without surface charge, the droplet would stay within the pore.
</p>

### Work plan

* Time dependent EHD
* Time dependent PF for the two-phase flow  
* Add the two above together
* More complicated geometries
* **Nobel Prize!!!** ***(only non-optional)***

### Folder plan
```
* BERNAISE
  * common
    * __init__.py
    * io.py
    * cmd.py
  * problems
    * __init__.py
    * simple.py
    * ...
  * solvers
    * __init__.py
    * basic.py
    * ...
  * tests
    * ...
  * meshes
    * ...
  * utilities
    * meshgeneration.py
    * meshload.py
    * ...
  * documentation
    * ...
  * articles
    * ...
  * README.md
  * sauce.py
```

### Master Minds: 
* **Gaute** er en ***nordmand***.
* **Asger** er ***dansk***.
* **Joachim** er *også* ***dansk***.
* **Alfred** *var* en ***svensker***.
