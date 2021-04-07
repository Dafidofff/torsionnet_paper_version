#!/usr/bin/env python

import sys
import os
import random
import numpy
from scm.plams import Molecule, DCDTrajectoryFile, dihedral
from crest import UniqueConformersMark
from scm.flexmd.structuraldescriptors.rmsd import fit_structure

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def read_conformers (mol, pathname, method='RDKit') :
        """
        Create a conformer set from files
        """
        dcdfilename = os.path.join(pathname,'%s.dcd'%(method))
        dcd = DCDTrajectoryFile(dcdfilename)
        nsteps = dcd.get_length()
        
        # Read in the corresponding energies
        enfilename = os.path.join(pathname,'energies%s.dat'%(method))
        infile = open(enfilename)
        lines = infile.readlines()
        infile.close()
        energies = []
        for line in lines :
                words = line.split() 
                if len(words) == 0 : continue
                energies.append(float(words[-1]))
        
        # Filter the conformers with Marks filter
        conformers = UniqueConformersMark()
        conformers.prepare_state(mol)
        for i,energy in enumerate(energies) :
                crd,cell = dcd.read_next()
                # Filtering is not necessary, since the conformers were already filtered
                duplicate = conformers.add_conformer(crd,energy)

        return conformers

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

def get_diffvec (phi,psi) :
        """
        Compute the shift
        """
        diff = phi-psi
        diff = diff - (numpy.round(diff/360)*360)
        return diff

def get_distance (phi,psi) :
        """
        Computes the difference between two dihedral angles
        """
        return abs(get_diffvec (phi,psi))

def get_average_torsion (phis) :
        """
        Get the average torsion angle, taking periodicity into account
        """
        shift = phis[0]
        phis_shifted = get_diffvec(phis,shift)
        avg_shifted = phis_shifted.sum()/len(phis)
        average = avg_shifted + shift
        return average

# Get the molecule
pathname = os.getcwd()
filename = os.path.join(pathname,'mol.xyz')
mol = Molecule(filename)
print ('nats: ',len(mol))

# Read in the conformers in this set
pathname = os.getcwd()
conformers = read_conformers (mol, pathname,method='ENUM')
nconfs = len(conformers)
print ('Original number of conformers: ',len(conformers))

# Read in the conformers in another, smaller set
pathname = '/home/bulo/Ravi/Molecules/set3/n_hexane/Enumerated120_strict/'
conformers_test = read_conformers (mol, pathname, method='ENUM')
print ('Number of conformers in subset: ',len(conformers_test))

# Combine the two sets into one big set (remembering which conformers came from which set)
all_conformers, (indices1, indices2) = conformers + conformers_test
print ('Total number of comformers: ',len(all_conformers))
print ('Indices test set in combined set: ',indices2)

# Which entries in the first set are not in the second set?
indices_new = [i for i,ind in enumerate(indices1) if not ind in indices2]
print ('\nContributions unique for the big set ')
print (len(indices_new), indices_new)
indices_orig = [i for i,ind in enumerate(indices1) if not ind in indices_new]
print ('\nRemaining contributions')
print (len(indices_orig), indices_orig)

##############
# Now find out what sets these new conformers apart from the rest
###############

# Define the torsion angles
torsions = sorted(define_rotatable_torsions(mol))
for tor in torsions: print(tor)
ntorsions = len(torsions)

# Get all the values for the torsion angles
torsion_angles = []
for iconf, crd in enumerate(conformers.geometries) :
        if iconf%100==0 : print (iconf)
        torsion_values = []
        for itors,atoms in enumerate(torsions) :
                phi = dihedral(*crd[atoms],unit='degree')
                torsion_values.append(phi)
        torsion_angles.append(torsion_values)
# Shape: (ntorsions,nconfs)
torsion_angles = numpy.array(torsion_angles).transpose()

# Create the distance matrix
matrices = []
for itors,phis in enumerate(torsion_angles) :
        ones = numpy.ones((nconfs,nconfs))
        matrix = get_distance(ones * phis.reshape((nconfs,1)), ones*phis.reshape((1,nconfs)) )

        # Get the minimum distance conformer for each conformer
        matrix[range(nconfs),range(nconfs)] = 1000.
        matrices.append(matrix)
# Shape: (ntorsions, nconfs, nconfs)
matrices = numpy.array(matrices)

# Now cut out the relevant part from the distance matrix
# Shape: (ntorsions, nconfs_new, nconfs_orig)
matrices = matrices[:,indices_new]
matrices = matrices[:,:,indices_orig]

# Now we need to get the average distance over the three torsion angles
# Shape: (nconfs_new, nconfs_orig)
avg_dists = matrices.sum(axis=0) / ntorsions

# Now we need for each of the new conformations the nearest index
index_mat = avg_dists.argsort(axis=1)
mindists = avg_dists.min(axis=1)

# Now print the result
for i, ind_n in enumerate(indices_new) :
        ind_o = indices_orig[index_mat[i][0]]
        phis_n = torsion_angles[:,ind_n]
        phis_o = torsion_angles[:,ind_o]
        mindist = mindists[i]
        print ('%3i %3i %6.1f '%(ind_n,ind_o,mindist),end='')
        for phi_n,phi_o in zip(phis_n,phis_o) :
                print ('[%6.1f %6.1f]'%(phi_n,phi_o),end='')
        print ()

# Find out why we consider these unique
nhat = len(conformers.trimmed_molecules[0])
i = 0
print ('Special angles: ',conformers.angles)
print ('Special dihedrals: ',conformers.changeable_dihedrals)
print ('Differences in distance matrix and torsion angles')
for i, ind_n in enumerate(indices_new) :
        ind_o = indices_orig[index_mat[i][0]]
        ddists, dtorsions = conformers.get_diffs_for_candidate(conformers.geometries[ind_n],ind_o)
        print ('%8i %8i %20.10f %20.10f '%(ind_n, ind_o,ddists[0],dtorsions[0]),end='')

        distance_matrix = conformers.distance_matrices[ind_n]
        indices = conformers.get_best_permutation(ind_o, distance_matrix)
        print (indices)
        best_mat = conformers.distance_matrices[ind_o][indices]
        best_mat = best_mat[:,indices]
        diff = abs(distance_matrix - best_mat)
        #print ('The atom pait that is most different (in the trimmed molecule)')
        #print (int(diff.argmax()/nhat), diff.argmax()%nhat)
