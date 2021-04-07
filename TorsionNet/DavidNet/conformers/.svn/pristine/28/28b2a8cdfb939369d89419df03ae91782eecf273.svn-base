#!/usr/bin/env python

import os
import numpy
import scipy.constants as sconst
import scipy.linalg
from scm.plams import Molecule
from scm.plams import DCDTrajectoryFile
from scm.plams import RKFTrajectoryFile
from scm.plams import Settings
from scm.flexmd.structuraldescriptors.rmsd import compute_rmsd
from .conformers_rotamers import ConformersRotamers

__all__ = ['UniqueConformersCrest']

class UniqueConformersCrest (ConformersRotamers) :
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
            >>> from scm.conformers import UniqueConformersCrest

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
                ConformersRotamers.__init__(self)

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
                #self.bconsts = []

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
                                # Also check if it is not a duplicate of the rotamers
                                isrot = self._is_not_rotamer_duplicate(coords, duplicate)
                        if isrot :
                                # Check the rmsd, and keep the smallest one in the conformer set
                                if self._is_rmsd_smaller(coords, duplicate) :
                                        coords, energy, bconsts = self._swap_conformers(duplicate, coords, energy, bconsts)
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

                ConformersRotamers.set_generator(self, method, engine_settings, nprocs_per_job, max_energy, nprocs)

        def generate (self, method='crest', nprocs_per_job=1, nprocs=1) :
                """
                Generate conformers using the specified method

                Note: Adjusts self

                * ``method`` -- A string, and one of the following options
                                ['crest', 'rdkit']
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS (only used if set_generator was not called)
                * ``nprocs``          -- Maximum number of parallel AMS processes ((only used if set_generator was not called))
                """
                ConformersRotamers.generate(self, method, nprocs_per_job, nprocs)

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
                if engine_settings is None :
                        engine_settings = Settings()
                        engine_settings.DFTB.Model = 'GFN1-xTB'
                ConformersRotamers.optimize(self,convergence_level,optimizer,max_energy,engine_settings,nprocs_per_job,nprocs,name)

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

        def read (self, dirname, name='crest', enfilename=None, reorder=True, check_for_duplicates=True, filetype='dcd') :
                """
                Read a conformer set from the specified directory
                """
                ConformersRotamers.read(self, dirname, name, enfilename, reorder, check_for_duplicates, filetype)

        def write (self, filename='crest', write_rotamers=False, dirname='.', filetype='dcd') :
                """
                Write the conformers to file
                """
                ConformersRotamers.write(self,filename,write_rotamers,dirname,filetype)

        def write_dcd (self, filename='crest', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in DCD format
                """
                ConformersRotamers.write_dcd(self,filename,write_rotamers,dirname)

        def write_rkf (self, filename='crest', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in RKF format
                """
                ConformersRotamers.write_rkf(self,filename,write_rotamers,dirname)

        def __str__ (self) :
                """
                Print conformer info
                """
                block = ConformersRotamers.__str__(self)
                lines = block.split('\n')
                newlines = [lines[0]+'%13s %13s %13s'%('Rot.Const.(1)','Rot.Const.(2)','Rot.Const.(3)')]
                for i,line in enumerate(lines[1:]) :
                        if len(line.split())==0 : continue
                        #bconst = numpy.sqrt((self.bconsts[i]**2).sum())
                        bconsts = self.bconsts[i]
                        newline = line + '%13.4f %13.4f %13.4f'%(bconsts[0],bconsts[1],bconsts[2])
                        newlines.append(newline)
                block = '\n'.join(newlines)
                return block

        # Private methods

        def _swap_conformers (self, iconformer, coords, energy, bconsts) :
                """
                Places the current conformer info in the set at position iconformer, 
                and returns the info from the previous iconformer
                """
                new_coords, new_energy = ConformersRotamers._swap_conformers(self, iconformer, coords, energy)
                new_bconsts = self.bconsts[iconformer].copy()
                self.bconsts[iconformer] = bconsts
                return new_coords, new_energy, new_bconsts

        def _add_if_not_duplicate (self, coords, energy, bconsts=[0.,0.,0.]) :
                """
                Add the candidate (check for duplicate has already been done)
                """
                ConformersRotamers._add_if_not_duplicate(self,coords,energy)
                self.bconsts.append(bconsts)

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

        def get_conformers_distmatrix (self) :
                """
                Produce a matrix representing the distances between conformers
                """
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
                scaling_factors = [1/energy_threshold] + 3*[1/bconst_treshold]
                vector = numpy.array([energy] + list(bconst)) * scaling_factors
                vector_j = numpy.array([self.energies[j]] + list(self.bconsts[j])) * scaling_factors
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
