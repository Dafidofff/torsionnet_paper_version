#!/usr/bin/env python

"""
Author: Mark Koenis, 2020

A class representing unique conformers defined by distance matrices and torsion angles.
"""

import os
import copy
import numpy
import itertools
from scm.plams import dihedral
from scm.plams import angle
from scm.plams import DCDTrajectoryFile
from scm.plams import Settings
from .conformers_rotamers import ConformersRotamers

__all__ = ['UniqueConformersAMS']

class UniqueConformersAMS (ConformersRotamers):
        """
        Class representing a set of unique conformers

        An instance of this class has the following attributes:

        *   ``molecule``    -- A PLAMS molecule object defining the connection data of the molecule
        *   ``geometries``  -- A list containing the coordinates of all conformers in the set
        *   ``energies``    -- A list containing the energies of all conformers in the set
        *   ``rotamers``    -- A list with UniqueConformersCrest objects representing the rotamer-set for each conformer
        *   ``copies``      -- A list containing the the times it was attempted to add each conformer to the set
        *   ``generator``   -- A conformer generator object. Has to be set with :meth:`set_generator`.
                               The default generator is of the CrestGenerator type.

        A simple example of (parallel) use::

            >>> from scm.plams import Molecule
            >>> from scm.plams import init, finish
            >>> from scm.conformers import UniqueConformersAMS

            >>> # Set up the molecular data
            >>> mol = Molecule('mol.xyz')
            >>> conformers = UniqueConformersAMS()
            >>> conformers.prepare_state(mol)

            >>> # Set up PLAMS settings
            >>> init()

            >>> # Create the generator and run
            >>> conformers.generate(nprocs_per_job=1, nprocs=12)

            >>> finish()

            >>> # Write the results to file
            >>> print(conformers)
            >>> conformers.write()

        The default generator for this conformer class is the RDKitGenerator.
        A list of all possibe generators:
        - RTDKitGenerator
        - TorsionGenerator
        - CrestGenerator
       
        By default the RDKitGenerator uses the UFF engine.
        To select a differen engine, set a different generator prior to runnung :meth:`generate`::

            >>> engine = Settings()
            >>> engine.ForceField.Type = 'UFF'
            >>> conformers.set_generator(method='rdkit', engine_settings=engine, nprocs_per_job=1, nprocs=12)

        The RDKitGenerator first uses RDKit to generate an initial set of conformer geometries.
        These are then subjected to geometry optimization using an AMS engine, 
        after which duplicates are filtered out.
        By default, the RDKitGenerator determines the number of initial conformers based on the
        number of rotational bonds in the system. 
        For a large molecule, this will result in a very large number of conformers.
        To set the number of initial conformers by hand, use::

            >>> conformers.set_generator(method='rdkit', nprocs_per_job=1, nprocs=12) 
            >>> conformers.generator.set_number_initial_conformers(100)
            >>> print ('Initial number of conformers: ',conformers.generator.ngeoms)
        """

        def __init__ (self, energy_threshold=0.2, min_dihed=30, min_dist=0.1) :
                """
                Creates an instance of the conformer class

                * ``energy_threshold`` -- kcal/mol
                * ``min_dist``         -- Maximum difference a distance between two atoms can have
                                          for a conformer to be considered a duplicate.
                * ``min_dihed``        -- Maximum difference a dihedral can have for a conformer to 
                                          be considered a duplicate. 
                """
                ConformersRotamers.__init__(self)

                # Thresholds used by Marks filter
                self.energy_threshold = energy_threshold
                self.min_dihed = min_dihed
                self.min_dist = min_dist

                # Info that needs to be stored for every conformer, to use Marks filter
                # Need to be manipulated in:
                #  - _add_it_not_duplicate()
                self.conformer_data['trimmed_molecules'] = [] # It's actually enough to just store the coordinates
                self.conformer_data['distance_matrices'] = []
                self.conformer_data['dihedral_values'] = []
                self.conformer_data['angle_values'] = []      # These are ordered as pairs matching the dihedral angles

                # General geometry info
                self.removable_atoms = []
                self.atomlabels = []          # A unique label for each atom
                self.changeable = []          # For each atom whether or not it is non-unique
                self.groups = []              # Indices of non-unique atoms grouped together
                self.fix = []                 # Unique atom ids
                self.dihedrals = []
                self.changeable_dihedrals = []
                self.angles = []              # Vervangt lineare dihedrals
                self.changeable_angles = []

        def prepare_state (self, mol, atoms_to_remove=None) :
                """
                Set up all the molecule data
                """
                ConformersRotamers.prepare_state(self, mol)

                # Remove the hydrogens
                self.removable_atoms = self.find_redundant_hydrogens(mol,atoms_to_remove)
                mol_trimmed = self.remove_hydrogens(mol=mol)
                mol_trimmed.guess_bonds()
                mol_trimmed.set_atoms_id()

                # Store geometry info
                self.atomlabels = self.get_linkage_network(mol_trimmed)
                changeable, groups, fix = self.find_interchangeable_atoms(mol_trimmed)
                self.changeable = changeable  # For each atom whether or not it is non-unique
                self.groups = groups          # Indices of non-unique atoms grouped together
                self.fix = fix                # Unique atom ids
                dihedrals = self.find_all_dihedrals(mol_trimmed)
                bad_dihedrals = self.find_unusable_dihedrals(dihedrals, mol_trimmed)
                # Remove dihedral angles for linear sections of the molecule
                self.dihedrals = [d for d in dihedrals if not d in bad_dihedrals]
                self.changeable_dihedrals = self.find_changeable_angles(self.dihedrals)

                # Replace those linear dihedral angles for two regular angles each
                #print ('dihedrals: ',len(dihedrals),dihedrals[0])
                #print ('bad_dihedrals: ',bad_dihedrals)
                tmp = [(dihedrals[ii][:3],dihedrals[ii][1:]) for ii in bad_dihedrals]
                self.angles = [angle for angles in tmp for angle in angles]
                        
                self.changeable_angles = self.find_changeable_angles(self.angles)

        def add_conformer (self, coords, energy=0., reorder=True, check_for_duplicates=True, accept_isomers=False) :
                """
                Adds a conformer to the list if it is not a duplicate

                Note: If the conformer is not unique, this returns the RMSD from its duplicate.
                      If it is unique, this returns the RMSD from the lowest energy geometry
                """
                # Check if valid coordinates are passed and if the conformer makes some sense
                # Also, shift to center of mass
                check = self._check_candidate(coords,energy,reorder,check_for_duplicates,accept_isomers)
                if not check : return None
                coords = self._translate_to_center_of_mass(coords)

                # Compute required data for candidate
                mol_trimmed = self.remove_hydrogens(coords=coords)
                distance_matrix = self.compute_distances(mol_trimmed)
                dihedral_values = self.compute_dihedrals(mol_trimmed)
                angle_values = self.compute_angles(mol_trimmed)

                duplicate = None
                if check_for_duplicates :
                        duplicate = self.find_duplicate(energy,distance_matrix,dihedral_values,angle_values)
                if duplicate is None :
                        self._add_if_not_duplicate(coords,energy,mol_trimmed,distance_matrix,dihedral_values,angle_values)
                        if reorder :
                                self.reorder()
                else :
                        isrot = False
                        if self.is_rotamer(duplicate, distance_matrix, dihedral_values, angle_values) :
                                # Also check if it is not a duplicate of the rotamers
                                isrot = self._is_not_rotamer_duplicate(coords, duplicate)
                        if isrot :
                                # Check the rmsd, and keep the smallest one in the conformer set
                                if self._is_rmsd_smaller(coords, duplicate) :
                                        conf = self._swap_conformers(duplicate,coords,energy,mol_trimmed,distance_matrix,dihedral_values,angle_values)
                                self.rotamers[duplicate]._add_if_not_duplicate(*conf)
                        else :
                                self.copies[duplicate] += 1
                        return duplicate

        def set_generator (self, method='rdkit', engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Store a generator object

                Note: Overwrites previous generator object

                * ``method``          -- A string, and one of the following options
                                         ['crest', 'rdkit']
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS
                * ``nprocs``          -- Maximum number of parallel AMS processes
                """
                # Prepare the engine
                if engine_settings is None :
                        engine_settings = Settings()
                        engine_settings.ForceField.Type = 'UFF'

                ConformersRotamers.set_generator(self, method, engine_settings, nprocs_per_job, energy_threshold, nprocs)

        def generate (self, method='rdkit', nprocs_per_job=1, nprocs=1) :
                """
                Generate conformers using the specified method

                Note: Adjusts self

                * ``method`` -- A string, and one of the following options
                                ['crest', 'rdkit']
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS (only used if set_generator was not called)
                * ``nprocs``          -- Maximum number of parallel AMS processes ((only used if set_generator was not called))
                """
                ConformersRotamers.generate(self, method, nprocs_per_job, nprocs)

        def get_diffs_for_candidate (self, coords, energy=0., iconf=None) :
                """
                Find out how much the values in the candidate molecule differ from each conformer
                """
                # Compare with which geometries
                confnums = [i for i in range(len(self.geometries))]
                if iconf is not None :
                        confnums = [iconf]

                mol_trimmed = self.remove_hydrogens(coords=coords)
                distance_matrix = self.compute_distances(mol_trimmed)
                dihedral_values = self.compute_dihedrals(mol_trimmed)
                angle_values = self.compute_angles(mol_trimmed)

                dist_diffs = []
                angle_diffs = []
                for j in confnums :
                        indices = self.get_best_permutation(j, distance_matrix, dihedral_values, angle_values)
                        max_difference = self.compare_dists_to_conformer(j, indices, distance_matrix)
                        max_angle_diff = self.compare_dihedrals_to_conformer(j, indices, dihedral_values, angle_values)
                        dist_diffs.append(max_difference)
                        angle_diffs.append(max_angle_diff)

                return dist_diffs, angle_diffs

        def read (self, dirname, name='ams', enfilename=None, reorder=True, check_for_duplicates=True, filetype='dcd') :
                """
                Read a conformer set from the specified directory
                """
                ConformersRotamers.read(self, dirname, name, enfilename, reorder, check_for_duplicates, filetype)

        def write (self, filename='ams', write_rotamers=False, dirname='.', filetype='dcd') :
                """
                Write the conformers to file
                """
                ConformersRotamers.write(self,filename,write_rotamers,dirname,filetype)

        def write_dcd (self, filename='ams', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in DCD format
                """
                ConformersRotamers.write_dcd(self,filename,write_rotamers,dirname)

        def write_rkf (self, filename='ams', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in RKF format
                """
                ConformersRotamers.write_rkf(self,filename,write_rotamers,dirname)

        # Private methods

        def _swap_conformers (self, iconformer, coords, energy, mol_trimmed, distance_matrix, dihedral_values, angle_value) :
                """
                Places the current conformer info in the set at position iconformer, 
                and returns the info from the previous iconformer
                """
                new_coords, new_energy = ConformersRotamers._swap_conformers(self, iconformer, coords, energy)

                new_mol_trimmed = self.trimmed_molecules[iconformer].copy()
                new_distance_matrix = self.distance_matrices[iconformer].copy()
                new_dihedral_values = self.dihedral_values[iconformer].copy()
                new_angle_value = self.angle_values[iconformer].copy()

                self.trimmed_molecules[iconformer] = mol_trimmed
                self.distance_matrices[iconformer] = distance_matrix
                self.dihedral_values[iconformer] = dihedral_values
                self.angle_values[iconformer] = angle_value
                return new_coords, new_energy, new_mol_trimmed, new_distance_matrix, new_dihedral_values, new_angle_value

        def _add_if_not_duplicate (self, coords, energy, mol_trimmed, distance_matrix, dihedral_values, angle_values) :
                """
                Add the candidate (check for duplicate has already been done)
                """
                ConformersRotamers._add_if_not_duplicate(self,coords,energy)
                self.trimmed_molecules.append(mol_trimmed)
                self.distance_matrices.append(distance_matrix)
                self.dihedral_values.append(dihedral_values)
                self.angle_values.append(angle_values)

        def find_duplicate (self, energy, distance_matrix, dihedral_values, angle_values) :
                """
                Checks if a certain coordinate/energy combination was already found

                Note: If a duplicate is found, the index of that duplicate is returned
                """
                duplicate = None
                for j in range(len(self.geometries)) :
                        # If the energy difference is large enough, this is not a duplicate
                        if abs(energy-self.energies[j]) >= self.energy_threshold: continue
                        indices = self.get_best_permutation(j, distance_matrix, dihedral_values, angle_values)
                        max_difference = self.compare_dists_to_conformer(j, indices, distance_matrix)
                        max_angle_diff = self.compare_dihedrals_to_conformer(j, indices, dihedral_values, angle_values)
                        if max_angle_diff < self.min_dihed and max_difference < self.min_dist:
                                duplicate = j
                                break
                return duplicate

        def is_rotamer (self, iframe, distance_matrix=None, coords=None, dihedral_values=None, angle_values=None) :
                """
                Checks if a the candidate is a rotamer of conformer iframe

                Note: The H-atoms are not included in this
                """
                if distance_matrix is None :
                        mol_trimmed = self.remove_hydrogens(coords=coords)
                        distance_matrix = self.compute_distances(mol_trimmed)
                        dihedral_values = self.compute_dihedrals(mol_trimmed)
                        angle_values = self.compute_angles(mol_trimmed)
                perm = self.get_best_permutation(iframe,distance_matrix,dihedral_values,angle_values)
                rotamer = perm != list(range(len(self.trimmed_molecules[0])))
                return rotamer

        def get_conformers_distmatrix (self) :
                """
                Produce a matrix representing the distances between conformers

                Note: Used in clustering methods (for which torsions are ignored)
                """
                ncomformers = len(self.geometries)
                dist_matrix = numpy.zeros((ncomformers,ncomformers))
                for i,dmat in enumerate(self.distance_matrices) :
                        for j in range(i+1,len(self.geometries)) :
                                indices = self.get_best_permutation(j, dmat)
                                dist = self.get_overlap_with_conformer(j,indices,dmat)
                                dist_matrix[i,j] = dist
                # Symmetrize
                dist_matrix = dist_matrix + dist_matrix.transpose()
                return dist_matrix

        def get_overlap_with_conformer (self, j, indices, distance_matrix) :
                """
                Computes an overlap value of conformer j and new candidate from the two distance matrices

                Note: tanimoto scoring function
                """
                dist_best = self.distance_matrices[j][indices]
                dist_best = dist_best[:,indices]
                inner_product = (distance_matrix*dist_best).sum()
                #normalizer = numpy.sqrt((distance_matrix**2).sum() * (dist_best**2).sum()) # This is what I would normalize with!
                normalizer = (distance_matrix**2).sum() + (dist_best**2).sum() - inner_product # This is what Mark normalizes with
                overlap = inner_product / normalizer
                #print ('overlap',j,overlap)
                overlap = numpy.sqrt(1-overlap)

                return overlap

        # Methods specific for the derived class

        def get_best_permutation (self, j, distance_matrix, dihedral_values=None, angle_values=None) :
                """
                Get best permutation of conformer j in self, for comparison with new candidate
                """
                def test_permutations (distance_matrix, groups, perms, indices, otherats, dist_matrix_j) :
                        """
                        Runs through permutations and compare distances to growing set of reference atoms 'otherats'
                        """
                        for g,group in enumerate(groups) :
                                differences = numpy.zeros(len(perms[g]))
                                max_diffs = []
                                for p, perm in enumerate(perms[g]) :
                                        # Swap atoms (rows first, then columns) in the new distance matrix
                                        indices_loc = indices[:]
                                        for gat,pat in zip(group,perm) :
                                                indices_loc[gat] = pat 
                                        dist_test = dist_matrix_j[indices_loc]
                                        dist_test = dist_test[:,indices_loc]
                                        # compute distance difference unique atoms
                                        diff_mat = distance_matrix[group][:,otherats] - dist_test[group][:,otherats]
                                        diff_test = numpy.absolute(diff_mat).sum()
                                        differences[p] = diff_test
                                # Store best permutations
                                best_perm = perms[g][differences.argmin()]
                                
                                # update otherats
                                for gat,bat in zip(group,best_perm) :
                                        indices[gat] = bat
                                otherats += group
                                otherats = sorted(otherats) # Maybe not necessary

                        return indices
        
                def print_matrix (diff_matrix) :
                        """
                        Print a matrix to standard out
                        """
                        for row in diff_matrix :
                                for v in row :
                                        print ('%6.2f '%(v),end='')
                                print()

                groups = copy.deepcopy(self.groups)
                perms = [list(itertools.permutations(group)) for group in groups]

                otherats = self.fix[:] # This is the set of reference atoms
                dist_matrix_j = copy.deepcopy(self.distance_matrices[j])
                indices = [i for i in range(len(dist_matrix_j))]

                # Find the best permutations of non-unique atoms for each group
                if len(otherats) > 0 :
                        indices = test_permutations(distance_matrix,groups,perms,indices,otherats,dist_matrix_j) 
                else :
                        # Because there are no distances to compare to for the permutations of the first atom-type
                        # they have to each be explicitly tried separately
                        ddist_list = []
                        max_ddists = []
                        indices_list = []
                        # Remove the first atom-type from the group
                        group = groups[0]
                        permutations_g0 = perms[0]
                        for perm in permutations_g0 :
                                # For each permutation of the first atom-type find the best permutation of the other atom-types
                                indices_tmp = indices[:]
                                for gat,pat in zip(group,perm) :
                                        indices_tmp[gat] = pat
                                otherats_tmp = sorted(otherats + group)
                                args = [distance_matrix,groups[1:],perms[1:],indices_tmp,otherats_tmp,dist_matrix_j]
                                indices_tmp = test_permutations(*args)
                                indices_list.append(indices_tmp)

                                # Compute a quality measure
                                dist_test = dist_matrix_j[indices_tmp]
                                dist_test = dist_test[:,indices_tmp]
                                diff_matrix = abs(dist_test - distance_matrix)
                                ddist_list.append( diff_matrix.sum() )
                                max_ddists.append( diff_matrix.max() )
                        # Compare the results for the permutations of the first atom-type
                        ddist_list = numpy.array(ddist_list)
                        indices = indices_list[ddist_list.argmin()]
                        # When there are several good options, an alternative approach is needed: Torsions
                        perm_inds = numpy.arange(len(permutations_g0),dtype=int)
                        good_perms = perm_inds[ numpy.array(max_ddists)<self.min_dist ]
                        if len(good_perms) > 0 :
                                if dihedral_values is not None and angle_values is not None :
                                        max_torsions = []
                                        for indices_tmp in indices_list :
                                                max_angle_diff = self.compare_dihedrals_to_conformer(j,indices_tmp,dihedral_values,angle_values)
                                                max_torsions.append(max_angle_diff)
                                        max_torsions = numpy.array(max_torsions)
                                        indices = indices_list[ good_perms[max_torsions[good_perms].argmin()] ]

                return indices

        def compare_dists_to_conformer (self, j, indices, distance_matrix) :
                """
                Find difference in dists between conformer j and new candidate
                """
                dist_best = self.distance_matrices[j][indices]
                dist_best = dist_best[:,indices]
                diff = abs(distance_matrix-dist_best)
                max_difference = diff.max()
                return max_difference

        def compare_dihedrals_to_conformer (self, j, indices, dihedral_values, angle_values) :
                """
                Find difference in dihedrals (or angles) between conformer j and new candidate
                """
                coords = self.trimmed_molecules[j].as_array()
                # First get dihedral angles
                dihedral_value2 = numpy.zeros(len(self.dihedrals))
                for ii,d in enumerate(self.dihedrals):
                        if self.changeable_dihedrals[ii] == 1:
                                dihedral_indices = [indices[i] for i in d]
                                # This angle must have already been stored, so it is a waste to recomupte it!
                                dihedral_value2[ii] = dihedral(*coords[dihedral_indices],unit='degree')
                        else:
                                dihedral_value2[ii] = self.dihedral_values[j][ii]
                diff = abs(dihedral_values-dihedral_value2)

                # Now get the angles
                angle_value2 = numpy.zeros(len(self.angles))
                for ii,a in enumerate(self.angles):
                        if self.changeable_angles[ii] == 1:
                                # This angle must have already been stored, so it is a waste to recomupte it!
                                angle_value2[ii] = angle(coords[a[0]]-coords[a[1]],coords[a[2]]-coords[a[1]],result_unit='degree')
                        else:
                                angle_value2[ii] = self.angle_values[j][ii]
                diff_a = abs(angle_values - angle_value2)

                # Combine angles and dihedrals and get the maximum 
                diff = numpy.concatenate((diff, diff_a))
                for d in range(len(diff)):
                        if diff[d] > 180:
                                diff[d] = 360 - diff[d]
                max_angle_diff = 0.
                if len(diff) > 0 :
                        max_angle_diff = max(diff)
                
                return max_angle_diff

        def find_redundant_hydrogens (self, mol, atoms_to_remove) :
                """
                Find the hydrogen atoms that can be removed

                Note: If atoms_to_remove is passed, then a default definition of removable H-atoms is used
                """
                if atoms_to_remove is not None :
                        return atoms_to_remove
                atom_remove = []
                # Finding non-essential H atoms
                for ii,atom in enumerate(mol) :
                        if atom.symbol == 'H':
                                neighbors = mol.neighbors(atom)
                                if len(neighbors) == 1 and neighbors[0].symbol == 'C':
                                        atom_remove.append(ii)
                return atom_remove

        def remove_hydrogens (self, mol=None, coords=None) :
                """
                Return the molecule without H
                """
                if mol is None :
                        mol_trimmed = self.molecule.copy()
                        mol_trimmed.from_array(coords)
                else :
                        mol_trimmed = mol.copy()
                for atom in reversed(self.removable_atoms): # start from end to keep index correct
                    mol_trimmed.delete_atom(mol_trimmed[atom+1])
                mol_trimmed.set_atoms_id()

                return mol_trimmed

        def get_linkage_network (self, mol_trimmed) :
                """
                Get the atom labels based on connectivity

                Note: This can be replaced by a single PLAMS command
                """
                #label = mol_trimmed.label(keep_labels=True)
                #atomlabels = [at.IDname for at in mol_trimmed.atoms]
                new_link = []
                atomlabels = []
                # Finding interchangable atoms
                for i,atom in enumerate(mol_trimmed):
                        connected = numpy.zeros(len(mol_trimmed))
                        connected[i] = 1
                        link_text = atom.symbol + ','
                        new_link.append(atom.id)
                        while len(new_link) > 0:
                                new_text = ''
                                new_link2 = []
                                for ii in new_link:
                                        neighbors = mol_trimmed.neighbors(mol_trimmed[ii])
                                        for atom1 in neighbors:
                                                if connected[atom1.id-1] == 0:  # a new atom was found
                                                        connected[atom1.id-1] = 1
                                                        new_link2.append(atom1.id)
                                                        new_text += atom1.symbol
                                new_link = new_link2
                                link_text += ''.join(sorted(new_text)) + ','
                        atomlabels.append(link_text)
        
                return atomlabels

        def find_interchangeable_atoms (self, mol_trimmed) :
                """
                Place all atoms that have the same label (interchangeable) in a group

                Note: changeable tells for each atom whether or not it has one or more identical copies
                """
                #atomlabels = numpy.array(self.atomlabels)
                #identicals = [numpy.where(atomlabels,label)[0] for label in set(atomlabels)]
                #identicals = [group for group in identicals if len(group)>1]        # groups
                #indices = [ind for group in groups for ind in group]
                #hasduplicate = [1 if i in indices else 0 for i in len(mol_trimmed)] # changeable
                #unique = numpy.where(changeable==0)[0]                              # fix

                changeable = numpy.zeros(len(mol_trimmed))
                groups = []
                for i,atom1 in enumerate(mol_trimmed):
                        group = []
                        if changeable[i] == 0:
                                for j,atom2 in enumerate(mol_trimmed):
                                        if i >= j : continue
                                        if changeable[j] == 0 and self.atomlabels[i] == self.atomlabels[j] :
                                                # these atoms are interchangable!
                                                if len(group) == 0:  # then it is a new group
                                                        group.append(i)
                                                        changeable[i] = 1
                                                group.append(j)
                                                changeable[j] = 1
                        if len(group) > 0:
                                groups.append(group)
                # Fix contains the index of each unique atom
                fix = list(numpy.arange(len(changeable))[changeable==0])

                return changeable, groups, fix

        def find_changeable_angles (self, angles) :
                """
                Find all changeable dihedrals in the molecule
                """
                changeable_angles = numpy.zeros(len(angles))
                for ii,angle in enumerate(angles):
                        for atom in angle:
                                if self.changeable[atom] == 1:  # this dihedral is changable
                                        changeable_angles[ii] = 1
                return changeable_angles

        def find_all_dihedrals(self, mol_trimmed):
                """
                this function searches all dihedral angles with bonded atoms within a
                molecule and returns them as a list.
                """
                dihedrals = []
                for i,atom1 in enumerate(mol_trimmed):
                        for atom2 in mol_trimmed.neighbors(mol_trimmed[atom1.id]):
                                j = atom2.id-1
                                for atom3 in mol_trimmed.neighbors(mol_trimmed[atom2.id]):
                                        k = atom3.id-1
                                        if atom1 == atom3 : continue
                                        for atom4 in mol_trimmed.neighbors(mol_trimmed[atom3.id]):
                                                l = atom4.id-1
                                                if atom2 == atom4 : continue
                                                dihedrals.append([i,j,k,l])
                # Filter out the duplicates (back and fro)
                double = []
                for i,d1 in enumerate(dihedrals) :
                        for j,d2 in enumerate(dihedrals) :
                                if i >= j : continue
                                if (d1[0] == d2[3] and d1[1] == d2[2] and d1[2] == d2[1] and d1[3] == d2[0]):
                                        double.append(d2)
                double.sort()
                dihedrals = [d for d in dihedrals if not d in double]
                return dihedrals

        def find_unusable_dihedrals(self,dihedrals, mol_trimmed):
                """
                Returns all dihedrals that are linear
                """
                coords = mol_trimmed.as_array()
                remove = []
                for ii,d in enumerate(dihedrals) :
                        a1 = angle(coords[d[0]]-coords[d[1]],coords[d[2]]-coords[d[1]],result_unit='degree')
                        a2 = angle(coords[d[1]]-coords[d[2]],coords[d[3]]-coords[d[2]],result_unit='degree')
                        if 170 <= a1 <= 190 or 170 <= a2 <= 190:  # bonds are nearly linear
                                remove.append(ii)
                return remove

        def compute_distances (self, mol) :
                """
                Compute all distances in this molecule object
                """
                nats = len(mol)
                ones = numpy.ones((nats,nats,3))
                matrix = (mol.as_array()*ones) - (mol.as_array().reshape((nats,1,3))*ones)
                dist_matrix = numpy.sqrt((matrix**2).sum(axis=2))

                return dist_matrix

        def compute_dihedrals (self, mol) :
                """
                Compute all dihedrals in themolecule object
                """ 
                coords = mol.as_array()
                dihedral_values = numpy.zeros(len(self.dihedrals))
                for ii,d in enumerate(self.dihedrals): #already callculate the reference dihedral angles
                        dihedral_values[ii] = dihedral(*coords[d],unit='degree')
                return dihedral_values

        def compute_angles (self, mol) :
                """
                Compute the subset of angles we need
                """
                coords = mol.as_array()
                angle_values = []
                for ii,a in enumerate(self.angles) :
                        angle1 = angle(coords[a[0]]-coords[a[1]],coords[a[2]]-coords[a[1]],result_unit='degree')
                        angle_values.append(angle1)
                angle_values = numpy.array(angle_values)
                return angle_values
