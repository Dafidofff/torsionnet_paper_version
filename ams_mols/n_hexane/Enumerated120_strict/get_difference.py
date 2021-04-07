#!/usr/bin/env python

import sys
from scm.plams import Molecule, DCDTrajectoryFile
from crest import UniqueConformersMark

#atoms = [int(w) for w in sys.argv[1:]]
atoms = [1,3]

mol = Molecule('mol.xyz')
dcd = DCDTrajectoryFile('ENUM.dcd')
conformers = UniqueConformersMark()
conformers.prepare_state(mol)
print ('Unique atoms: ',conformers.fix)
print ('identical atoms: ',conformers.groups)

crd,cell = dcd.read_frame(atoms[0])
duplicate = conformers.add_conformer(crd)

crd,cell = dcd.read_frame(atoms[1])
ddists, dtors = conformers.get_diffs_for_candidate(crd)

print ('Distance: ',ddists[0])
print ('Torsion:  ',dtors[0])

# Look closer
duplicate = conformers.add_conformer(crd)

if duplicate is None :
        distance_matrix = conformers.distance_matrices[1]
        print ('Now testing the permutation code')
        indices = conformers.get_best_permutation(0, distance_matrix)
        print (indices)
