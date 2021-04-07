#!/usr/bin/env python

import sys
import numpy
import os
from scm.plams import Molecule, DCDTrajectoryFile, get_conformations, JobRunner
from scm.plams import init, finish, config
from crest import UniqueConformersMark, MoleculeOptimizer, get_rotatable_bonds

dirname = sys.argv[1] # cyclohexane
nconformers = None
if len(sys.argv) > 2 :
        nconformers = int(sys.argv[2])

# Read in the molecule for which comformers will be generated
filename = os.path.join(dirname,'mol.xyz')
mol = Molecule(filename)
nrotbonds = int(get_rotatable_bonds(mol))  # For cyclohexane I set this to 6
if nconformers is None :
        nconformers = 2 * (3**nrotbonds)   # For cyclohexane I did not use the factor 2
molnum = 0
print ('nats: ',len(mol))
print ('nrotbonds: ',nrotbonds)
print ('nconformers: ',nconformers)

# Change into working directory
os.chdir(dirname)

# Generate conformers
mol.guess_bonds()
bondorder = mol.bond_matrix()
mat = numpy.matrix(bondorder)
conformers = get_conformations(mol, nconformers, enforceChirality=True)
#conformers = get_conformations(mol, 10, enforceChirality=True)
geometries = [plmol.as_array() for plmol in conformers]

# Optimize the geometries of the conformers (GFN1-xTB is the default)
init(folder='Molecule%03i'%(molnum))
# Run in parallel
config.default_jobrunner = JobRunner(parallel=True, maxjobs=16)
optimizer = MoleculeOptimizer(mol,nproc=2)
geometries, energies = optimizer.optimize_geometries(geometries)
finish()

# Filter out the duplicates
conformers = UniqueConformersMark()
conformers.prepare_state(mol)
for crd, energy in zip(geometries, energies) :
        # Try to add this to the conformer object. It will be filtered.
        #print (conformers.get_diffs_for_candidate (crd, energy))
        duplicate = conformers.add_conformer(crd,energy)
print ('Number of conformers after Marks filtering',len(conformers))
print ('%20s %20s'%('Conformer','#Rotamers'))
for i in range(len(conformers)) :
        print ('%20i %20i'%(i,len(conformers.rotamers[i])))

# Write the resulting conformers to file (and the energies)
dcd = DCDTrajectoryFile('RDKit.dcd',mode='wb',ntap=len(mol))
for i,crd in enumerate(conformers.geometries) :
        dcd.write_next(coords=crd)
dcd.close()
outfile = open('energiesRDKit.dat','w')
for i,energy in enumerate(conformers.energies) :
        outfile.write('%8i %20.10f\n'%(i,energy))
outfile.close()

# This is an example of writing the rotamers to file (may be too many files)
for i in range(len(conformers)) :
        if len(conformers.rotamers[i].geometries) == 0 : continue
        dcd = DCDTrajectoryFile('rotamers%i.dcd'%(i),mode='wb',ntap=len(mol))
        for i,crd in enumerate(conformers.rotamers[i].geometries) :
                dcd.write_next(coords=crd)
        dcd.close()

# Change back home
os.chdir('../')
