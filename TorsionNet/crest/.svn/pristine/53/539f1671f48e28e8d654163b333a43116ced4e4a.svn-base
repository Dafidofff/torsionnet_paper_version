#!/usr/bin/env python

import os
import numpy
import scipy.constants as sconst
import scipy.linalg
from scm.plams import Molecule
from scm.plams import DCDTrajectoryFile
from scm.plams import Settings
from scm.flexmd.structuraldescriptors.rmsd import compute_rmsd
from .conformers import Conformers

__all__ = ['UniqueConformersCrest']

class UniqueConformersCrest (Conformers) :
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
            >>> from crest import UniqueConformersCrest

            >>> # Set up the molecular data
            >>> mol = Molecule('mol.xyz')
            >>> conformers = UniqueConformersCrest()
            >>> conformers.prepare_state(mol)

            >>> # Set up PLAMS settings
            >>> init()

            >>> # Create the generator and run
            >>> conformers.generate(nprocs_per_job=1, nprocs=12)

            >>> finish()

            >>> # Write the results to file
            >>> print(conformers)
            >>> conformers.write()

        Note: The default generator for this conformer class is the CrestGenerator, using the GFN1-xTB engine.
              This will generally take a lot of time.
              To speed things up, set a different generator prior to runnung :meth:`generate`::

            >>> engine = Settings()
            >>> engine.ForceField.Type = 'UFF'
            >>> conformers.set_generator(method='crest', engine_settings=engine, nprocs_per_job=1, nprocs=12)
        """

        def __init__ (self, energy_threshold=0.1, rmsd_threshold=0.125, bconst_treshold=15.) :
                """
                Creates an instance of the conformer class

                * ``energy_threshold`` -- Energy in kcal/mol
                * ``rmsd_threshold``   -- RMSD in Angstrom
                * ``bconst_treshold``  -- Rotational constant in MHz
                """
                Conformers.__init__(self)

                # Thresholds used
                self.energy_threshold = energy_threshold
                self.rmsd_threshold = rmsd_threshold
                self.bconst_treshold = bconst_treshold

                # Info that needs to be stored for every conformer
                # Need to be manipulated in:
                #  - _add_it_not_duplicate()
                #  - reorder()
                #  - remove_conformer()
                self.conformer_data['bconsts'] = []
                self.conformer_data['rotamers'] = []
                #self.bconsts = []
                #self.rotamers = []

        def get_all_geometries (self) :
                """
                Get all the gemetries in the set
                """
                geometries = self.geometries
                geometries += [crds for rotamers in self.rotamers for crds in rotamers.geometries]
                return geometries

        def add_conformer (self, coords, energy, reorder=True, check_for_duplicates=True, accept_isomers=False) :
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

                # Compute the rotational constants
                bconsts= self.get_Bconst(coords)

                duplicate = None
                if check_for_duplicates :
                        duplicate = self.find_duplicate(energy,coords,bconsts)
                if duplicate is None :
                        self._add_if_not_duplicate(coords,energy,bconsts)
                        if reorder :
                                self.reorder()
                else :
                        isrot = False
                        if self.is_rotamer(duplicate, coords) :
                                isrot = True
                                # Also check if it is not a duplicate of the rotamers
                                for iframe in range(len(self.rotamers[duplicate])) :
                                        if not self.rotamers[duplicate].is_rotamer(iframe, coords) :
                                                isrot = False
                        if isrot :
                                #self.rotamers[duplicate].add_conformer(coords, energy)
                                self.rotamers[duplicate]._add_if_not_duplicate(coords,energy,bconsts)
                        else :
                                self.copies[duplicate] += 1
                return duplicate

        def set_generator (self, method='crest', engine_settings=None, nprocs_per_job=1, max_energy=6., nprocs=1) :
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
                        engine_settings.DFTB.Model = 'GFN1-xTB'

                Conformers.set_generator(self, method, engine_settings, nprocs_per_job, max_energy, nprocs)

        def generate (self, method='crest', nprocs_per_job=1, nprocs=1) :
                """
                Generate conformers using the specified method

                Note: Adjusts self

                * ``method`` -- A string, and one of the following options
                                ['crest', 'rdkit']
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS (only used if set_generator was not called)
                * ``nprocs``          -- Maximum number of parallel AMS processes ((only used if set_generator was not called))
                """
                Conformers.generate(self, method, nprocs_per_job, nprocs)

        #def remove_conformer (self, index) :
        #        """
        #        Remove a conformer from the set
        #        """
        #        Conformers.remove_conformer(self,index)
        #        self.bconsts = [b for i,b in enumerate(self.bconsts) if i!=index]
        #        self.rotamers = [r for i,b in enumerate(self.rotamers) if i!=index]

        def get_diffs_for_candidate (self, coords, energy, iconf=None) :
                """
                Find out how much the values in the candidate molecule differ from each conformer
                """
                # Compare with which geometries
                confnums = [i for i in range(len(self.geometries))]
                if iconf is not None :
                        confnums = [iconf]

                bconsts= self.get_Bconst(coords)

                energy_diffs = []
                bconst_diffs = []
                for j in confnums :
                        energy_diffs.append(energy - self.energies[j])
                        bconst_diffs.append( numpy.sqrt(((bconsts-self.bconsts[j])**2).sum()) )
                        #bconst_diffs.append( abs((bconsts-self.bconsts[j])).max() ) # REB

                return energy_diffs, bconst_diffs

        def read (self, dirname, name='crest', enfilename=None, reorder=True, check_for_duplicates=True) :
                """
                Read a conformer set from the specified directory
                """
                Conformers.read(self, dirname, name, enfilename, reorder, check_for_duplicates)

        def write (self, filename='crest', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file
                """
                Conformers.write(self,filename,dirname)

                if write_rotamers :
                        for i in range(len(self)) :
                                pathname = os.path.join(dirname,'rotamers_%s%i.dcd'%(filename,i))
                                dcd = DCDTrajectoryFile(pathname,mode='wb',ntap=len(self.molecule))
                                for i,crd in enumerate(self.rotamers[0].geometries) :
                                        dcd.write_next(coords=crd)
                                dcd.close()

        def __str__ (self) :
                """
                Print conformer info
                """
                block = Conformers.__str__(self)
                lines = block.split('\n')
                newlines = [lines[0]+'%20s %13s %13s %13s'%('#Rotamers','Rot.Const.(1)','Rot.Const.(2)','Rot.Const.(3)')]
                for i,line in enumerate(lines[1:]) :
                        if len(line.split())==0 : continue
                        #bconst = numpy.sqrt((self.bconsts[i]**2).sum())
                        bconsts = self.bconsts[i]
                        newline = line + '%20i %13.4f %13.4f %13.4f'%(len(self.rotamers[i]),bconsts[0],bconsts[1],bconsts[2])
                        newlines.append(newline)
                block = '\n'.join(newlines)
                return block

        # Private methods

        def _add_if_not_duplicate (self, coords, energy, bconsts=[0.,0.,0.]) :
                """
                Add the candidate (check for duplicate has already been done)
                """
                Conformers._add_if_not_duplicate(self,coords,energy)
                self.bconsts.append(bconsts)
                self.rotamers.append(UniqueConformersCrest())
                self.rotamers[-1].prepare_state(self.molecule)

        def find_duplicate (self, energy, coords, bconsts) :
                """
                Checks if a certain coordinate/energy combination was already found
                """
                duplicate = None
                for iframe,crd in enumerate(self.geometries) :
                        de = abs(energy - self.energies[iframe])
                        db = abs(bconsts - self.bconsts[iframe])
                        rmsd_db = numpy.sqrt(sum(db**2))
                        #rmsd_db = db.max()  # REB
                        if de < self.energy_threshold and rmsd_db < self.bconst_treshold:
                                duplicate = (iframe) #duplicates
                                break
                return duplicate

        def is_rotamer (self, iframe, coords) :
                """
                Checks if a certain coordinate/energy combination was already found
                """
                rotamer = False
                rmsd = self.get_rmsd(coords, iframe)
                if rmsd > self.rmsd_threshold :
                        rotamer = True
                return rotamer

        #def reorder (self) :
        #        """
        #        Reorder conformers from smallest to largest energy
        #        """
        #        indices = Conformers.reorder(self)
        #        self.bconsts = [self.bconsts[i] for i in indices]
        #        self.rotamers = [self.rotamers[i] for i in indices]
        #        return indices

        def get_conformers_distmatrix (self) :
                """
                Produce a matrix representing the distances between conformers
                """
                ncomformers = len(self.geometries)
                for i,(en,b) in enumerate(zip(self.energies,self.bconsts)) :
                        for j in range(i+1,len(self.geometries)) :
                                dist = self.get_overlap_with_conformer(j,en,b)
                                dist_matrix[i,j] = dist
                # Symmetrize
                dist_matrix = dist_matrix + dist_matrix.transpose()
                return dist_matrix

        def get_overlap_with_conformer (self, j, energy, bconst) :
                """
                Computes an overlap value of conformer j and new candidate from the two distance matrices

                Note: tanimoto scoring function
                """
                scaling_factors = [1/energy_threshold, 1/bconst_treshold]
                vector = numpy.array([energy, bconst]) * scaling_factors
                vector_j = numpy.array([self.energies[j],self.bconsts[j]]) * scaling_factors
                inner_product = (vector*vector_j).sum()
                #normalizer = numpy.sqrt((vector**2).sum() * (vector_j**2).sum()) # This is what I would normalize with!
                normalizer = (vector**2).sum() + (vector_j**2).sum() - inner_product # This is what Mark normalizes with
                overlap = inner_product / normalizer
                #print ('overlap',j,overlap)
                overlap = numpy.sqrt(1-overlap)

                return overlap

        # Methods specific for the derived class

        def get_Bconst (self, coords, units='MHz') :
                """
                Get the rotational constant (B) for all conformers. 

                * ``units``    -- Can either be 'cm-1' or 'MHz'

                Note: To compare to a database of computed rotational constants
                      https://cccbdb.nist.gov/rotcalc2x.asp
                """
                coords = self._translate_to_center_of_mass(coords)
                masses = numpy.array([atom.mass for atom in self.molecule])
                nats = len(masses)

                tensor_comps = numpy.zeros((nats,3,3))
                for k in range(3):
                        for l in range(3):
                                if k == l:
                                        p = (k+1) %3 
                                        q = (k+2) %3 
                                        tensor_comps[:,k,k] += masses * (coords[:,p]**2 + coords[:,q]**2)
                                else:   
                                        tensor_comps[:,k,l] += -1 * masses * coords[:,k] * coords[:,l]
                i_tens = tensor_comps.sum(axis=0)

                eigvals, eigvecs = scipy.linalg.eig(i_tens)
                eigvals = eigvals.real

                # Convert from (g/mol)*A2 to kg*m*s-1
                eigvals *= 1.e-23/sconst.N_A

                # Compute rotational constant in s-1 (correct for linear molecules)
                eigvals_tmp = eigvals.copy()
                eigvals_tmp[eigvals==0.] = 1.
                bconsts = sconst.h / (8*sconst.pi**2 * eigvals_tmp)
                bconsts[eigvals==0.] = 0.

                if units=='cm-1' :
                        # Convert to from s-1 to cm-1
                        bconsts = bconsts*1.e-2 / sconst.c
                elif units=='MHz' :
                        # Convert from s-1 to MHz
                        bconsts *= 1e-6
                else :
                        raise Exception('The unit %s is not recognized for rotational constants '%(units))
                bconsts = numpy.sort(bconsts)[::-1]
        
                return bconsts

        def get_rmsd (self, coords, i) :
                """
                Compute RMSD with respect to conformer i
                """
                rmsd, grad = compute_rmsd(self.geometries[i],coords,compute_grad=False)
                return rmsd

        def get_rmsds_from_frame (self, frame) :
                """
                Get all RMSDs from a certain frame
                """
                coords = self.geometries[frame]

                rmsds = []
                for iframe,crd in enumerate(self.geometries) :
                        rmsd, grad = compute_rmsd(crd,coords,compute_grad=False)
                        rmsds.append(rmsd)
                return rmsds
