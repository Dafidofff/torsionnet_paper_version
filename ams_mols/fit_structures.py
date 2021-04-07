#!/usr/bin/env python

import os
import shutil
import numpy
from scm.plams import Molecule, DCDTrajectoryFile
from scm.flexmd.structuraldescriptors.rmsd import fit_structure
from scm.flexmd import pdb_from_plamsmol, MDMolecule, locate_rings

def get_atoms_smallest_ring (mol) :
        """
        Get the atoms in the smallest ring to be used to overlay all conformers
        """
        # Get the atoms in small rings, as they can be expected to be most rigid
        pdb = pdb_from_plamsmol(mol)
        mdmol = MDMolecule(pdb=pdb)
        rings = locate_rings(mdmol,[i for i in range(len(mol))])
        rings = [ring for ring in rings if len(ring) <= 6]
        atoms = []
        if len(rings) > 0 :
                ind = [len(ring) for ring in rings].index(min([len(ring) for ring in rings]))
                atoms = rings[ind]
        # If there are no rings, use all carbon atoms
        if len(atoms) == 0 : atoms = [i for i in range(len(mol)) if mol.atoms[i].symbol=='C']
        return atoms

dirname = 'cyclohexane'
molname = os.path.join(dirname,'mol.xyz')

mol = Molecule(molname)

# Get the atoms in small rings, as they can be expected to be most rigid
atoms = get_atoms_smallest_ring (mol)

# Move the dcd file to a different file, and overwrite the original one
dcdfilename = os.path.join(dirname,'RDKit.dcd')
shutil.move(dcdfilename, os.path.join(dirname,'RDKit_orig.dcd'))
dcd = DCDTrajectoryFile(os.path.join(dirname,'RDKit_orig.dcd'))
coords,_ = dcd.read_frame(1)

# Now do the fitting and the writing
outdcd = DCDTrajectoryFile(dcdfilename,mode='wb',ntap=dcd.ntap)
for i in range(len(dcd)) :
        print (i)
        crd,cell = dcd.read_frame(i)
        crd_new = fit_structure(numpy.array(crd),mol.as_array(),atoms)
        outdcd.write_next(coords=crd_new)
