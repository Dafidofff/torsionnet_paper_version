#!/usr/bin/env python

import sys
import os
import copy
import numpy
import scipy
from scipy import signal
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scm.plams import Molecule, DCDTrajectoryFile, get_conformations, dihedral, RKFTrajectoryFile, KFFile, Units
from scm.flexmd import pdb_from_plamsmol
from crest import UniqueConformersMark, MoleculeOptimizer, get_rotatable_bonds

"""
The goal is to find out for each bond if all three torsion angles are represented
"""

def define_rotatable_torsions (mol) :
        """
        Return the atoms of the torsion angles for the rotatable bonds
        """
        torsions = []
        for bond in mol.bonds :
                indices = [ind-1 for ind in mol.index(bond)]
                terminal = False
                identical = False
                for i,at in enumerate(bond) :
                        neighbors = at.neighbors()
                        if len(neighbors)<4 :
                                terminal = True
                                continue
                        labels = set([at.IDname for at in neighbors if not at in bond])
                        if len(labels) == 1 :
                                identical = True
                if terminal or identical :
                        continue
                # Define the torsion
                one_four = []
                for at in bond :
                        one_four += [mol.index(n)-1 for n in at.neighbors() if n.symbol=='C' and not n in bond]
                torsion = [one_four[0]] + indices + [one_four[1]]
                torsions.append(torsion)  

        return torsions

#################
en_conv = 1.e-7 # default 1.e-5
grad_conv = 1.e-5 # default 1.e-3
dtheta = 30
nrotations = int(360/dtheta)
##############

mol = Molecule('mol.xyz')
mol.guess_bonds()
mol.label(keep_labels=True)
# Define the torsion angles
torsions = sorted(define_rotatable_torsions(mol))

# Read all optimized geometries and energies from the plams folder
plamsdirname = 'Molecule000'
dirnames = os.listdir(plamsdirname)
dirnames = sorted([os.path.join(plamsdirname,dn) for dn in dirnames if not 'logfile' in dn])
energies = []
geometries = []
for dn in dirnames :
        print (dn)
        filename = os.path.join(dn,'ams.rkf')
        rkf = RKFTrajectoryFile(filename)
        crd,cell = rkf.read_last_frame()
        nsteps = len(rkf)
        kf = KFFile(filename)
        energy = kf.read('History','Energy(%i)'%(nsteps)) * Units.conversion_ratio('Hartree','kcal/mol')
        energies.append(energy)
        geometries.append(copy.deepcopy(crd))

# Filter out the duplicates
conformers = UniqueConformersMark()
conformers.prepare_state(mol)
for i,(crd, energy) in enumerate(zip(geometries, energies)) :
        # Try to add this to the conformer object. It will be filtered.
        #print (conformers.get_diffs_for_candidate (crd, energy))
        duplicate = conformers.add_conformer(crd,energy)
        if duplicate is not None :
                print ('duplicate: %8i %8i'%(i, duplicate))
print ('Number of conformers after Marks filtering',len(conformers))
print ('%20s %20s'%('Conformer','#Rotamers'))
for i in range(len(conformers)) :
        print ('%20i %20i'%(i,len(conformers.rotamers[i])))

# Print the torsion angles
for i,crd in enumerate(conformers.geometries) :
        dih = []
        for tor in torsions :
                dih.append(dihedral(*crd[tor],unit='degree'))
        print ('%8i %10.1f %10.1f %10.1f'%(i,dih[0],dih[1],dih[2]))

# Write the resulting conformers to file (and the energies)
dcd = DCDTrajectoryFile('ENUM.dcd',mode='wb',ntap=len(mol))
for i,crd in enumerate(conformers.geometries) :
        dcd.write_next(coords=crd)
dcd.close()
outfile = open('energiesENUM.dat','w')
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
