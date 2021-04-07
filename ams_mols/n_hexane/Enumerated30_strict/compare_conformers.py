#!/usr/bin/env python

import sys
import os
import random
import numpy
from scm.plams import Molecule, DCDTrajectoryFile
from crest import UniqueConformersMark

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
                duplicate = conformers.add_conformer(crd,energy,check_for_duplicates=False)

        return conformers

# Get the molecule
pathname = os.getcwd()
filename = os.path.join(pathname,'mol.xyz')
mol = Molecule(filename)
print ('nats: ',len(mol))

# Read in the conformers you have
pathname = os.getcwd()
conformers = read_conformers (mol, pathname,method='ENUM')
print ('Original number of conformers: ',len(conformers))

# Create a random subset of conformers, just so that we have something to compare with
pathname = '/home/bulo/Ravi/Molecules/set3/n_hexane/Enumerated60_strict/'
conformers_test = read_conformers (mol, pathname, method='ENUM')
print ('Number of conformers in subset: ',len(conformers_test))

# Check the lowest energy of the testset
print('\n###########################')
e_diff = conformers_test.energies[0] - conformers.energies[0]
print ('Lowest relative energy: %20.10f kcal/mol'%(e_diff))
print('###########################\n')
print ('Quality lowest energy conformers (between 0 and 1): ',numpy.exp(-e_diff))

# Combine the two sets into one big set (remembering which conformers came from which set)
all_conformers, (indices1, indices2) = conformers + conformers_test
print ('Total number of comformers: ',len(all_conformers))
print ('Indices test set in combined set: ',indices2)
names = all_conformers.indices_to_names(indices1, indices2)

# Print the percentage of conformers found by the testset
print('\n###########################')
print ('Fraction of conformers found: ',len(conformers_test)/len(all_conformers))
print('###########################\n')

# Make a dendrogram plot for the big set
dendrogram = all_conformers.get_dendrogram()
figure = all_conformers.get_plot_dendrogram(dendrogram, names=names, fontsize=8)
figure.savefig('dendro.jpg')

# Now assign conformers to clusters
cluster_indices1, cluster_indices2 = all_conformers.find_clusters(0.11,'distance',indices=(indices1, indices2))
print (cluster_indices1)
print (cluster_indices2)
nclusters = max(cluster_indices1+cluster_indices2)
print('nclusters: ',nclusters)
print('\n###########################')
print ('Fraction of clusters in testset: ',len(set(cluster_indices2))/nclusters)
print('###########################')

