#!/usr/bin/env python

import os
import numpy
import scipy.constants as sconst
import scipy.linalg
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import TorsionFingerprints
from scm.plams import Molecule
from scm.plams import DCDTrajectoryFile
from scm.plams import RKFTrajectoryFile
from scm.plams import Settings
from scm.flexmd.structuraldescriptors.rmsd import compute_rmsd
from .conformers import Conformers

__all__ = ['UniqueConformersTFD']

class UniqueConformersTFD (Conformers) :
        """
        Class representing a set of unique conformers

        An instance of this class has the following attributes:

        *   ``molecule``    -- A PLAMS molecule object defining the connection data of the molecule
        *   ``rdmol``       -- RDKit molecule object without conformers
        *   ``geometries``  -- A list containing the coordinates of all conformers in the set
        *   ``energies``    -- A list containing the energies of all conformers in the set
        *   ``copies``      -- A list containing the the times it was attempted to add each conformer to the set
        *   ``generator``   -- A conformer generator object. Has to be set with :meth:`set_generator`.
                               The default generator is of the RDKitGenerator type.

        A simple example of (parallel) use::

            >>> from scm.plams import Molecule
            >>> from scm.plams import init, finish
            >>> from scm.conformers import UniqueConformersTFD

            >>> # Set up the molecular data
            >>> mol = Molecule('mol.xyz')
            >>> conformers = UniqueConformersTFD()
            >>> conformers.prepare_state(mol)

            >>> # Set up PLAMS settings
            >>> init()

            >>> # Create the generator and run
            >>> conformers.generate(nprocs_per_job=1, nprocs=12)

            >>> finish()

            >>> # Write the results to file
            >>> print(conformers)
            >>> conformers.write()

        Note: The default generator for this conformer class is the RDKitGenerator, using the GFN1-xTB engine.
              This will generally take a lot of time.
              To speed things up, set a different generator prior to runnung :meth:`generate`::

            >>> engine = Settings()
            >>> engine.ForceField.Type = 'UFF'
            >>> conformers.set_generator(method='rdkit', engine_settings=engine, nprocs_per_job=1, nprocs=12)
        """

        def __init__ (self, tfd_threshold=0.05) :
                """
                Creates an instance of the conformer class

                * ``tfd_threshold``   -- Torsion Fingerprint (unitless)
                """
                Conformers.__init__(self)

                # Thresholds used
                self.tfd_threshold = tfd_threshold

                # Info that needs to be stored for every conformer
                self.conformer_data['torsions'] = []

                # Data specific for the derived class
                self.rdmol = None
                self.torsion_list = None
                self.ringtorsion_list = None
                self.torsion_weights = None
                self.use_weights = True

        def prepare_state(self, mol) :
                """
                Set up all the molecule data
                """
                Conformers.prepare_state(self,mol)
                self.rdmol = self._get_empty_rdkitmol()

                torsions = TorsionFingerprints.CalculateTorsionLists(self.rdmol)
                self.torsion_list = torsions[0]
                self.ringtorsion_list = torsions[1]

                # Get the weights for the torsion angles
                self.torsion_weights = []
                if len(self.torsion_list + self.ringtorsion_list) > 0 :
                        self.torsion_weights = TorsionFingerprints.CalculateTorsionWeights(self.rdmol)

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
                torsions = self._compute_torsion_angles(coords)

                duplicate = None
                if check_for_duplicates :
                        duplicate = self.find_duplicate(energy,coords,torsions)
                if duplicate is None :
                        self._add_if_not_duplicate(coords,energy,torsions)
                        if reorder :
                                self.reorder()
                else :
                        # Keep only the one with the lowest energy
                        swapped = False
                        if energy < self.energies[duplicate] :
                                coords,energy,torsions = self._swap_conformers(duplicate,coords,energy,torsions)
                                swapped = True
                        self.copies[duplicate] += 1
                        if swapped and reorder :
                                self.reorder()
                return duplicate

        def set_generator (self, method='rdkit', engine_settings=None, nprocs_per_job=1, max_energy=6., nprocs=1) :
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

                Conformers.set_generator(self, method, engine_settings, nprocs_per_job, max_energy, nprocs)

        def generate (self, method='rdkit', nprocs_per_job=1, nprocs=1) :
                """
                Generate conformers using the specified method

                Note: Adjusts self

                * ``method`` -- A string, and one of the following options
                                ['crest', 'rdkit']
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS (only used if set_generator was not called)
                * ``nprocs``          -- Maximum number of parallel AMS processes ((only used if set_generator was not called))
                """
                Conformers.generate(self, method, nprocs_per_job, nprocs)

        def get_diffs_for_candidate (self, coords, energy, iconf=None) :
                """
                Find out how much the values in the candidate molecule differ from each conformer
                """
                # Compare with which geometries
                confnums = [i for i in range(len(self.geometries))]
                if iconf is not None :
                        confnums = [iconf]

                torsions = self._compute_torsion_angles(coords)

                energy_diffs = []
                tfds = []
                for j in confnums :
                        weights = self.torsion_weights
                        if not self.use_weights :
                                weights = None
                        tfd = TorsionFingerprints.CalculateTFD(torsions,self.torsions[j],weights=weights)
                        tfds.append(tfd)
                        energy_diffs.append(energy - self.energies[j])

                #tfds = self._get_tfds_for_candidate(coords,confnums)

                return energy_diffs, tfds

        def read (self, dirname, name='tfd', enfilename=None, reorder=True, check_for_duplicates=True, filetype='dcd') :
                """
                Read a conformer set from the specified directory
                """
                Conformers.read(self, dirname, name, enfilename, reorder, check_for_duplicates, filetype)

        def write (self, filename='tfd', dirname='.', filetype='dcd') :
                """
                Write the conformers to file
                """
                Conformers.write(self,filename,dirname,filetype)

        def write_dcd (self, filename='tfd', dirname='.') :
                """
                Write the conformers to file in DCD format
                """
                Conformers.write_dcd(self,filename,dirname)

        def write_rkf (self, filename='tfd', dirname='.') :
                """
                Write the conformers to file in RKF format
                """
                Conformers.write_rkf(self,filename,dirname)

        def __str__ (self) :
                """
                Print conformer info
                """
                block = Conformers.__str__(self)
                lines = block.split('\n')
                newlines = [lines[0]+'%13s'%('TFD')]
                for i,line in enumerate(lines[1:]) :
                        if len(line.split())==0 : continue
                        tfd = self.get_tfd_between_frames(0,i)
                        newline = line + '%13.4f'%(tfd)
                        newlines.append(newline) 
                block = '\n'.join(newlines)
                return block

        # Private methods

        def _swap_conformers (self, iconformer, coords, energy, torsions) :
                """
                Places the current conformer info in the set at position iconformer, 
                and returns the info from the previous iconformer
                """
                new_coords, new_energy = ConformersRotamers._swap_conformers(self, iconformer, coords, energy)
                new_torsions = self.torsions[iconformer].copy()
                self.torsions[iconformer] = torsions
                return new_coords, new_energy, new_torsions

        def _add_if_not_duplicate (self, coords, energy, torsions) :
                """
                Add the candidate (check for duplicate has already been done)
                """
                Conformers._add_if_not_duplicate(self,coords,energy)
                self.torsions.append(torsions)

        def find_duplicate (self, energy, coords, torsions) :
                """
                Checks if a certain coordinate/energy combination was already found
                """
                duplicate = None
                for iframe,coords in enumerate(self.geometries) :
                        weights = self.torsion_weights
                        if not self.use_weights :
                                weights = None
                        tfd = TorsionFingerprints.CalculateTFD(torsions,self.torsions[iframe],weights=weights)
                        if tfd < self.tfd_threshold:
                                duplicate = (iframe) #duplicates
                                break
                return duplicate

        def get_conformers_distmatrix (self) :
                """
                Produce a matrix representing the distances between conformers
                """
                for i,en in enumerate(self.energies) :
                        #tfds = self._get_tfds_for_candidate(self.geometries[i])
                        for j in range(i+1,len(self.geometries)) :
                                tfd = self.get_tfd_between_frames(i,j)
                                dist_matrix[i,j] = tfd # tfds[j]
                # Symmetrize
                dist_matrix = dist_matrix + dist_matrix.transpose()
                return dist_matrix

        # Methods specific for the derived class

        def get_torsion_atoms (self) :
                """
                Returns all the torsion atoms involved in the TFD

                Note:   Each contribution is a list of sets of four atoms. Mostly the list has only one entry,
                        but in case of symmetry, more sets of 4 atoms can contribute to a single torsion value.
                """
                torsion_atoms = [t[0] for t in self.torsion_list]
                torsion_atoms += [t[0] for t in self.ringtorsions_list]
                return torsion_atoms

        def get_torsion_values (self, iconf) :
                """
                Get the values of all the torsion angles for this conformer

                Note: Each contribution is a list of torion angles. Mostly the list has only one entry,
                      but in the case of symmetry, or rings, several torsion angles contribute to a single TFP value.
                """
                torsions = [t[0] for t in self.torsions]
                return torsions

        def get_tfd_between_frames (self, iconf, jconf) :
                """
                Compute the TFD between frames
                """
                weights = self.torsion_weights
                if not self.use_weights :
                        weights = None
                return TorsionFingerprints.CalculateTFD(self.torsions[iconf],self.torsions[jconf],weights=weights)

        def _get_tfds_for_candidate (self, coords, confnums=None) :
                """
                Get the TFDs for this candidate frim all other conformers
                """
                if confnums is None :
                        confnums = [i for i in range(len(self))]

                self.rdmol = self._add_conformers_to_rdkitmol(self.rdmol)
                # Create the new conformer and add it to thr RDKit molecule
                conf = Chem.Conformer()
                for i, crd in enumerate(coords):
                        xyz = Geometry.Point3D(crd[0], crd[1], crd[2])
                        conf.SetAtomPosition(i, xyz)
                self.rdmol.AddConformer(conf, assignId=True)
                #print ('ConfIDs: ',[c.GetId() for c in self.rdmol.GetConformers()])
                # Compare TFDs
                tfds = TorsionFingerprints.GetTFDBetweenConformers(self.rdmol,confnums, [len(self)],useWeights=self.use_weights)
                self.rdmol.RemoveAllConformers()
                return tfds

        def _compute_torsion_angles (self, coords) :
                """
                Compute the torsion angle values for coordinates
                """
                conf = self._get_rdkit_conformer(coords)
                self.rdmol.AddConformer(conf)
                args = (self.rdmol,self.torsion_list,self.ringtorsion_list,0)
                torsions = TorsionFingerprints.CalculateTorsionAngles(*args)
                self.rdmol.RemoveAllConformers()
                return torsions

