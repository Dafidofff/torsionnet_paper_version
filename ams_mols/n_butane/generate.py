#!/usr/bin/env python

import numpy
from scm.plams import Molecule, DCDTrajectoryFile, get_conformations, JobRunner
from scm.plams import init, finish, config
from crest import UniqueConformersMark, MoleculeOptimizer

# Read in the molecule for which comformers will be generated
molnum = 2
filename = 'mol.xyz'
mol = Molecule(filename)
print ('nats: ',len(mol))
nconfs = 50

# Generate conformers
mol.guess_bonds()
bondorder = mol.bond_matrix()
mat = numpy.matrix(bondorder)
conformers = get_conformations(mol, nconfs, enforceChirality=True)
geometries = [plmol.as_array() for plmol in conformers]

# Filter out the duplicates
conformers = UniqueConformersMark()
conformers.prepare_state(mol,atoms_to_remove=[])
print ('State prepared')
for i,crd in enumerate(geometries) :
        print (i)
        # Try to add this to the conformer object. It will be filtered.
        #print (conformers.get_diffs_for_candidate (crd, energy))
        duplicate = conformers.add_conformer(crd,0.)
        if duplicate is not None :
                print ('%8i %8i'%(i,duplicate))
print ('Number of conformers after Marks filtering',len(conformers))
print ('%20s %20s'%('Conformer','#Rotamers'))
for i in range(len(conformers)) :
        print ('%20i %20i'%(i,len(conformers.rotamers[i])))

