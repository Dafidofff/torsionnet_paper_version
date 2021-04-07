#!/usr/bin/env python

import numpy
import scipy
from scipy import signal
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scm.plams import Molecule, DCDTrajectoryFile, dihedral

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

def check_clusters (centers, threshold=30.) :
        """
        Check that the clusters are approximately 120 degrees apart
        """
        ones = numpy.ones(nclusters)
        distances = get_distance(ones*centers.reshape((nclusters,1)), ones*centers.reshape((1,nclusters)))
        distances = distances[numpy.triu(distances)!=0]
        return (abs(distances-120)).max() <= threshold

mol = Molecule('mol.xyz')
mol.guess_bonds()
mol.label(keep_labels=True)
nats = len(mol)

dcd = DCDTrajectoryFile('RDKit.dcd')

# Define the torsion angles
torsions = define_rotatable_torsions(mol)
for tor in torsions: print(tor)

# Get all the values for the torsion angles
coords = mol.as_array()
torsion_angles = {}
for iconf,(crd,cell) in enumerate(dcd) :
        if iconf%100==0 : print (iconf)
        coords[:] = crd
        for itors,atoms in enumerate(torsions) :
                phi = dihedral(*coords[atoms],unit='degree')
                if not itors in torsion_angles :
                        torsion_angles[itors] = []
                torsion_angles[itors].append(phi)
for itors,atoms in enumerate(torsion_angles) :
        torsion_angles[itors] = numpy.array(torsion_angles[itors])

# Now we need to cluster them somehow (for a function over the histogram and find maxima?
confnum = 6 # The conformation I want to compare
ntorsions = len(torsions)
size_list = []
dists_to_conf = []
for itors in range(ntorsions) :
        phis = torsion_angles[itors]
        #print ('phis: ',phis)

        # Compute a distance matrix
        nconfs = len(phis)
        ones = numpy.ones((nconfs,nconfs))
        matrix = get_distance(ones * phis.reshape((nconfs,1)), ones*phis.reshape((1,nconfs)) )

        # Get the minimum distance conformer for each conformer
        print ('Matrix ',itors)
        print (matrix[confnum,:10])
        matrix[range(nconfs),range(nconfs)] = 1000.
        dists_to_conf.append(matrix[confnum])

dists_to_conf = numpy.array(dists_to_conf)

# Now find the conformers that is overall the most similar
dist_to_conf = dists_to_conf.sum(axis=0) / ntorsions
indices = dist_to_conf.argsort()
mindist = dist_to_conf.min() 
print ()
print ('The nearest conformer: ',indices[:5], mindist)
