#!/usr/bin/env python

import os
import numpy
import scipy.linalg
from scm.plams import Molecule
from scm.plams import DCDTrajectoryFile
from scm.plams import Settings
from scm.plams.mol.identify import twist
from scm.flexmd.structuraldescriptors.rmsd import compute_rmsd
from scm.flexmd.structuraldescriptors.rmsd import fit_structure
from ..optimizer import MoleculeOptimizer
from ..generators.generator import Generator
from ..generators.crest_generators.crest_generator import CRESTGenerator
from ..generators.simple_generators.rdkit_generator import RDKitGenerator
from ..generators.simple_generators.torsion_generator import TorsionGenerator

__all__ = ['Conformers','get_stereochemistry','get_cis_trans']

class Conformers :
        """
        Abstract Class representing a set of conformers
        """

        def __init__ (self) :
                """
                Creates an instance of the conformer class

                Note: To be added to in derived classes
                """
                self.molecule = None
                self.label = None
                self.fit_atoms = None

                # Need to be manipulated in:
                #  - _add_if_not_duplicate()
                #  - reorder()
                #  - remove_conformer()
                # This should be a dictionary. Then those methods can loop over it automatically
                self.conformer_data = {}
                self.conformer_data['geometries'] = []
                self.conformer_data['energies'] = []
                self.conformer_data['copies'] = []
                #self.geometries = []
                #self.energies = []
                #self.copies = []

                # Generator object
                self.generator = None

        def prepare_state (self, mol) :
                """
                Set up all the molecule data

                Note: To be added to in derived classes
                """
                # Create shortcuts to conformer_data content
                self._create_variables()

                if len(self.geometries) > 0 :
                        raise Exception('Cannot prepare state after adding geometries')
                self.molecule = mol
                self.molecule.guess_bonds()
                self.label = self.molecule.label(keep_labels=True)
                # if there is not ring it will return the full backbone
                self.fit_atoms = self.molecule.get_atoms_smallest_ring() 

        def _create_variables (self) :
                """
                Create instance variables out of the content of the conformer_data dictionary

                Note: Since all elements of the dictionary are a list,
                      they are all mutable, and so the new instance variables
                      should point to the same data as the dictionary items
                """
                for key,item in self.conformer_data.items() :
                        self.__dict__[key] = self.conformer_data[key]
           
        def copy (self) :
                """
                Copy the conformer set

                Note: Used as is in all derived classes
                """
                newconfset = self.__class__()
                for var in self.__dict__.keys() :
                        try :
                                newconfset.__dict__[var] = self.__dict__[var].copy()
                        except AttributeError :
                                newconfset.__dict__[var] = self.__dict__[var]
                return newconfset

        def __len__ (self) :
                """
                Determines the number of conformers in this set
                """
                return len(self.geometries)

        def __str__ (self) :
                """
                Print conformer info
                """
                block = '%20s %20s %20s\n'%('Conformer','Energy','NCopies')
                min_energy = min(self.energies)
                for i in range(len(self)) :
                        block += '%20i %20.6f %20i\n'%(i,self.energies[i]-min_energy,self.copies[i])
                return block

        def __add__ (self, conformers) :
                """
                Add the second conformer set to the first, and return as new

                Note: Used as is in all derived classes
                """
                newconfset = self.copy()

                numlist1 = [i for i in range(len(self.geometries))]
                numlist2 = []
                for i,(en,crds) in enumerate(zip(conformers.energies, conformers.geometries)) :
                        num = newconfset.add_conformer(crds,en,reorder=False)
                        if num is None :
                                num = len(newconfset.geometries)-1
                        numlist2.append(num)

                # Now create names
                #names = newconfset.indices_to_names(numlist1,numlist2)

                return newconfset, (numlist1,numlist2)

        def get_all_geometries (self) :
                """
                Get all the gemetries in the set
                """
                return self.geometries

        def get_energies (self) :
                """
                Returns the energies in reference to the most stable

                Note: Used as is in all derived classes
                """
                return [en-self.energies[0] for en in self.energies]

        def get_all_rmsds (self) :
                """
                Get the RMSD value from the lowest energy conformer for all conformers

                Note: Used as is in all derived classes
                """
                if len(self.geometries) > 0 :
                        coords = self.geometries[0]
                rmsds = []
                for iframe,crd in enumerate(self.geometries) :
                        rmsd, grad = compute_rmsd(crd,coords,compute_grad=False)
                        rmsds.append(rmsd)
                return rmsds
        rmsds = property(get_all_rmsds)

        def add_conformer (self, coords, energy, reorder=True, check_for_duplicates=True, accept_isomers=False) :
                """
                Adds a conformer to the list if requirements are met

                Note: Adds every conformer
                """
                # Check if valid coordinates are passed and if the conformer makes some sense
                # Also, shift to center of mass
                check = self._check_candidate(coords,energy,reorder,check_for_duplicates,accept_isomers)
                if not check : return None
                coords = self._translate_to_center_of_mass(coords)

                duplicate = None
                # This will always return None
                if check_for_duplicates :
                        duplicate = self.find_duplicate(energy,coords)
                if duplicate is None :
                        self._add_if_not_duplicate(coords,energy)
                        if reorder :
                                self.reorder()
                return duplicate

        def set_generator (self, method, engine_settings, nprocs_per_job=1, max_energy=6., nprocs=1) :
                """
                Store a generator object

                Note: Overwrites previous generator object

                * ``method``          -- A string, and one of the following options
                                         ['crest', 'rdkit','torsion']
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS
                * ``nprocs``          -- Maximum number of parallel AMS processes
                """
                # Do a check
                if self.molecule is None :
                        raise Exception ('State has to be prepared first!')
                
                # Prepare the generator and generate
                if method == 'crest' :
                        self.generator = CRESTGenerator(self, engine_settings, nprocs_per_job, max_energy, nprocs)
                elif method == 'rdkit' :
                        self.generator = RDKitGenerator(self, engine_settings, nprocs_per_job, max_energy, nprocs)
                elif method == 'torsion' :
                        self.generator = TorsionGenerator(self, engine_settings, nprocs_per_job, max_energy, nprocs)
                else :
                        raise Exception('Method %s does not exist'%(method))

        def generate (self, method, nprocs_per_job=1, nprocs=1) :
                """
                Generate conformers using the specified method

                Note: Adjusts self

                * ``method`` -- A string, and one of the following options
                                ['crest', 'rdkit','torsion']
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS (only used if set_generator was not called)
                * ``nprocs``          -- Maximum number of parallel AMS processes ((only used if set_generator was not called))
                """
                if self.generator is None :
                        self.set_generator(method,nprocs_per_job=nprocs_per_job,nprocs=nprocs)
                else :
                        method = self.generator.name
                if self.generator.name != method :
                        print ('WARNING: Generator used (%s) does not match specified method %s'%(self.generator.name,method))
                #self.generator.prepare_state()
                self.generator.generate()

        def optimize (self, convergence_level, optimizer=None, max_energy=None, engine_settings=None, nprocs_per_job=1, nprocs=1, name='go') :
                """
                (Re)-Optimize the conformers currently in the set

                * ``convergence_level`` -- One of the convergence options ('tight', 'vtight', 'loose', etc')
                * ``optimizer``         -- Instance of the MoleculeOptimizer class. 
                                           If not provided, an engine_settings object is required.
                * ``engine_settings``   -- PLAMS Settings object:
                                           engine_settings = Settings()
                                           engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                # Collect all geometries (possibly including rotamers)
                geometries = self.get_all_geometries()

                if optimizer is None :
                        if engine_settings is None :
                                raise Exception ('Either an MoleculeOptimizer object or an engine_settings object is required')
                        optimizer = MoleculeOptimizer(self.molecule,engine_settings,nprocs_per_job,nprocs)

                convergence = Generator.convergence_settings[convergence_level]
                geometries, energies = optimizer.optimize_geometries(geometries, convergence, name)

                # Changing self 
                self.clear()
                min_energy = min([en for en in energies if not en is None])
                for i,(crd,en) in enumerate(zip(geometries,energies)) :
                        if crd is None : continue
                        if max_energy is not None :
                                if (en-min_energy) >= max_energy :
                                        continue
                        self.add_conformer(crd,en)

        def clear (self) :
                """
                Remove all conformers
                """
                for key, values in self.conformer_data.items() : 
                        self.conformer_data[key] = []
                self._create_variables()

        def remove_conformer (self, index) :
                """
                Remove a conformer from the set

                Note: To be added to in derived classes
                """
                #self.geometries = [crd for i,crd in enumerate(self.geometries) if i!=index] 
                #self.energies = [en for i,en in enumerate(self.energies) if i!=index]
                #self.copies = [c for i,c in enumerate(self.copies) if i!=index]

                for key, values in self.conformer_data.items() :
                        self.conformer_data[key] = [v for i,v in enumerate(values) if i!=index]
                self._create_variables()

        def get_diffs_for_candidate (self, coords, energy, iconf=None) :
                """
                Find out how much the values in the candidate molecule differ from each conformer

                Note: To be specified in derived classes
                """
                # Compare with which geometries
                confnums = [i for i in range(len(self.geometries))]
                if iconf is not None :
                        confnums = [iconf]
                return ([None]*len(confnums))

        def find_clusters (self, dist=5., criterion='maxclust', method='average', indices=None) :
                """
                Assign all conformers to clusters

                Note: Used as is in all derived classes

                :parameter float dist:    Either the max number of clusters (for maxclust), 
                                          or the maximum distance between clusters (for distance)
                :parameter str criterion: Determines how many clusters to make.
                                          Mark prefers to set the number of clusters (maxclust)
                                          I would prefer to set the maximum distance between clusters (distance)
                :parameter tuple indices: A tuple with as elements lists of indices for subsets of conformers
                """
                from scipy.cluster.hierarchy import fcluster

                dend = self.get_dendrogram(method)
                cluster_indices = fcluster(dend, dist, criterion=criterion)
                if indices is None :
                        return cluster_indices
                else :
                        all_cluster_indices = []
                        for indlist in indices :
                                cinds = [ind for i,ind in enumerate(cluster_indices) if i in indlist]
                                all_cluster_indices.append(cinds)
                        return tuple(all_cluster_indices)

        def get_dendrogram (self, method='average') :
                """
                Gets a dendrogram reflecting the distances between conformers

                Note: Used as is in all derived classes
                """
                from scipy.spatial.distance import squareform
                from scipy.cluster.hierarchy import linkage

                distmatrix = self.get_conformers_distmatrix()
                dend = linkage(squareform(distmatrix), method)
                return dend

        def get_plot_dendrogram (self, dend, names=None, fontsize=4) :
                """
                Makes a plot of the dendrogram

                Note: Used as is in all derived classes
                """
                from scipy.cluster.hierarchy import dendrogram
                from scipy.cluster import hierarchy
                from matplotlib import pyplot as plt

                # Set up some default names for the conformers
                if names is None :
                        names = ['%i'%(i) for i in range(len(self.geometries))]

                # Create the matplotlib figure, and make it the desirable size
                fig = plt.figure()
                fig.set_size_inches(6, len(dend)/15+2)

                # Add color palette
                hierarchy.set_link_color_palette(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']) 

                # Create the axis object, and make sure the x-axis has the correct fontsize
                ax = fig.add_subplot(111)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)

                # Add the dendrogram plot to the axis object
                dendrogram(dend, labels=names,  orientation='right', leaf_font_size=fontsize, ax=ax)

                # Shift along x-axis, so that (near)duplicates are visible
                ax.set_xlim([-0.001, None])

                return fig

        def indices_to_names (self, indices1, indices2, name1='a',name2='b') :
                """
                Convert two sets of indices to names for the conformers in self

                Note: - Used as is in all derived classes
                      - Mostly for use related to clustering features
                      - Only works with two sets of indices.
                      - All indices need to be represented by these two lists
                """
                names = []
                for i in range(len(self.geometries)) :
                        if i in indices1 and i in indices2 :
                                ind1 = indices1.index(i)
                                ind2 = indices2.index(i)
                                #names.append('a%02ib%02i'%(ind1,ind2))
                                names.append(name1+'%02i -- '%(ind1)+name2+'%02i'%(ind2))
                        elif i in indices1 and not i in indices2 :
                                ind = indices1.index(i)
                                names.append(name1+'%02i'%(ind))
                        elif i not in indices1 and i in indices2 :
                                ind = indices2.index(i)
                                names.append(name2+'%02i'%(ind))
                        else :
                                raise Exception ('All conformers have to be present in either one of the indexlists')
                return names

        def read (self, dirname, name='base', enfilename=None, reorder=True, check_for_duplicates=True) :
                """
                Read a conformer set from the specified directory
                """
                if enfilename is None :
                        enfilename = 'energies_%s.txt'%(name)

                filename = os.path.join(dirname,enfilename)
                infile = open(filename)
                lines = infile.readlines()
                infile.close()
                
                energies = []
                for line in lines :
                        words = line.split()
                        if len(words) == 0 : continue
                        energy = float(words[1])
                        energies.append(energy)

                dcdfilename = os.path.join(dirname,'%s.dcd'%(name))
                dcd = DCDTrajectoryFile(dcdfilename)
                
                for i,(crd,cell) in enumerate(dcd) :
                        self.add_conformer(crd.copy(),energies[i],reorder,check_for_duplicates)

        def write (self, filename='base', dirname='.') :
                """
                Write the conformers to file
                """
                # Write the resulting conformers to file (and the energies)
                pathname = os.path.join(dirname,'%s.dcd'%(filename))
                dcd = DCDTrajectoryFile(pathname,mode='wb',ntap=len(self.molecule))
                for i,crd in enumerate(self.geometries) :
                        dcd.write_next(coords=crd)
                dcd.close()
                enpathname = os.path.join(dirname,'energies_%s.txt'%(filename))
                outfile = open(enpathname,'w')
                for i,energy in enumerate(self.energies) :
                        outfile.write('%8i %20.10f\n'%(i,energy))
                outfile.close()

        # Private methods

        def _check_candidate (self, coords, energy, reorder, check_for_duplicates, accept_isomers) :
                """
                Check if the candidate is reasonable
                """
                # Check if valid coordinates are passed
                if coords is None :
                        print ('Warning: No coordinates passed')
                        return False

                # Check this candidate conformer makes some sense
                if not (self.is_candidate_valid(coords, accept_isomers)) :
                        print ('Warning: The new molecule is not the same as the first molecule')
                        if len(self.geometries)==0 :
                                print ('The lowest energy conformer is kept')
                                print ('If the order of the atoms has changed, this may go wrong')
                                self.keep_best_conformer(coords, energy, reorder)
                        return False

                return True

        def _translate_to_center_of_mass (self,coords) :
                """
                Translate coordinates to center of mass
                """
                nats = len(self.molecule)
                masses = numpy.array([atom.mass for atom in self.molecule])
                com = (coords*masses.reshape((nats,1))).sum(axis=0) / masses.sum()
                coords = coords - com
                return coords

        def _add_if_not_duplicate (self, coords, energy) :
                """
                Add the candidate (check for duplicate has already been done)

                Note: To be added to in the derived classes
                """
                # Find the best fit of the coordinates to the first one in the set.
                if len(self) == 0 :
                        coords_fitted = coords.copy()
                else :
                        coords_fitted = fit_structure(coords,self.geometries[0],self.fit_atoms)
                self.geometries.append(coords_fitted)
                #self.geometries.append(coords)
                self.energies.append(energy)
                self.copies.append(1)

        def find_duplicate (self, energy, coords) :
                """
                Checks if a certain coordinate/energy combination was already found

                Note: To be specified in the derived classes
                """
                return None

        def reorder (self) :
                """
                Reorder conformers from smallest to largest energy

                Note: To be added to in the derived classes
                """
                indices = numpy.array(self.energies).argsort()
                #self.energies = [self.energies[i] for i in indices]
                #self.geometries = [self.geometries[i] for i in indices]
                #self.copies = [self.copies[i] for i in indices]
                for key, values in self.conformer_data.items() :
                        self.conformer_data[key] = [values[i] for i in indices]
                self._create_variables()

                return indices

        def is_candidate_valid (self, coords, accept_isomers=False) :
                """
                Checks that the candidate is the same as the rest of the set

                Note: Used as is in all derived classes
                """
                # Produce a unique label for the molecules already in the set
                rs_ref = []
                ct_ref = []
                if not accept_isomers :
                        dic = get_stereochemistry(self.molecule, label_present=True)
                        keys = sorted([k for k in dic.keys()])
                        rs_ref = [dic[k] for k in keys]
                        dic = get_cis_trans(self.molecule, label_present=True)
                        ct_keys = sorted([k for k in dic.keys()])
                        ct_ref = [dic[k] for k in ct_keys]

                # Produce a unique label for the candidate, by creating a new plams molecule
                plmol = self.get_plamsmol_from_coords (coords)
                if plmol is None :
                        return False
                plmol.guess_bonds()
                label = plmol.label(keep_labels=True)
                rs = []
                ct = []
                if not accept_isomers :
                        dic = get_stereochemistry(plmol, label_present=False)
                        keys = sorted([k for k in dic.keys()])
                        rs = [dic[k] for k in keys]
                        dic = get_cis_trans(plmol, label_present=True)
                        ct_keys = sorted([k for k in dic.keys()])
                        ct = [dic[k] for k in ct_keys]
                #print (keys, rs_ref, rs)
                return label==self.label and rs==rs_ref and ct==ct_ref

        def keep_best_conformer (self, coords, energy, reorder) :
                """
                Keeps the conformer with the lowest energy

                Note: Used as is in all derived classes
                """
                if len(self) != 1 :
                        # No conformers should be thrown away if there are more than 1!
                        return
                plmol = self.get_plamsmol_from_coords (coords)
                if energy < energy[0] and plmol is not None :
                        self.remove_conformer(0)
                        self.prepare_state(plmol)
                        self.add_conformer(coords, energy, reorder)
                
        def get_plamsmol_from_coords (self, coords) :
                """
                Produce a new plamsmol for a candidate

                Note: Used as is in all derived classes
                """
                elements = [at.symbol for at in self.molecule]
                plmol = Molecule.from_elements(elements)
                if len(coords) is not len(elements) :
                        return None
                plmol.from_array(coords)
                return plmol
                              
        def _names_to_indices (self, names) :
                """
                Convert a list of names for all conformers to two lists of indices

                Note: - Used as is in all derived classes
                      - Mostly for use related to clustering features

                names:  List of strings, each representing the name of a conformer
                        Each name has the form 'aiibjj', 'aii', or 'bjj', with ii and jj being an integer number.
                """
                na = len([1 for name in names if 'a' in name])
                nb = len([1 for name in names if 'b' in name])
                indices1 = numpy.zeros(na)
                indices2 = numpy.zeros(nb)
                for i,name in enumerate(names) :
                        if 'a' in name :
                                ind = int(name.split('a')[1].split('b')[0])
                                indices1[ind] = i
                        if 'b' in name :
                                ind = int(name.split('b')[1])
                                indices2[ind] = i
                return indices1, indices2

        def get_conformers_distmatrix (self) :
                """
                Produce a matrix representing the distances between conformers

                Note: To be specified in the derived classes
                """
                n = len(self.geomstries)
                return numpy.zeros((n,n))

        def get_overlap_with_conformer (self, j, energy, bconst) :
                """
                Computes an overlap value of conformer j and new candidate from the two distance matrices

                Note: To be specified in the derived classes
                """
                return 0

def get_stereochemistry (mol, label_present=False) :
        """
        Get the info about the chiral centers
        """
        # It is important to use labels without stereochemistry here
        if not label_present :
                mol.label(keep_labels=True)
        # Now look for atoms with four different substituents
        directions = {}
        others  = {}
        for atom in mol.atoms :
                neighbors = mol.neighbors(atom)
                id_neighbors  = sorted([mol.index(n)-1 for n in neighbors])
                neighbors = [mol.atoms[i] for i in id_neighbors]
                if len(neighbors) < 4 : continue
                idnames = set([at.IDname for at in neighbors])
                if len(idnames) >= 4 :
                        vecs = [atom.vector_to(n) for n in neighbors[:3]]
                        sign, other = twist(*vecs)
                        iat = mol.index(atom)-1
                        directions[iat] = sign
                        others[iat] = other
        return directions

def get_cis_trans (mol, label_present=False) :
        """
        Get the info about cis/trans isomers
        """
        if not label_present :
                mol.label(keep_labels=True)
        isomerization = {}
        for bond in mol.bonds :
                if bond.order != 2 : continue
                double_bond = True
                for atom in bond :
                        neighbors = mol.neighbors(atom)
                        if len(neighbors) != 3 : double_bond = False
                        idnames = set([at.IDname for at in neighbors])
                        if len(idnames) < 3 : double_bond = False
                if not double_bond : continue
                pvecs = []
                for atom in bond :
                        neighbors = mol.neighbors(atom)
                        id_neighbors  = sorted([mol.index(n)-1 for n in neighbors])
                        neighbors = [mol.atoms[i] for i in id_neighbors]
                        vecs = [atom.vector_to(n) for n in neighbors[:3]]
                        uvecs = [v/numpy.linalg.norm(v) for v in vecs]
                        pvecs.append(numpy.cross(uvecs[0],uvecs[1]))

                iats = tuple([mol.index(atom)-1 for atom in bond])
                iso = int(numpy.sign((pvecs[0]*pvecs[1]).sum()))
                isomerization['%2i-%2i'%(iats[0],iats[1])] = iso
        return isomerization
