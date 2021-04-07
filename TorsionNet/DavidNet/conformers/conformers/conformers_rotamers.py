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
from .conformers import Conformers

__all__ = ['ConformersRotamers']

class ConformersRotamers (Conformers) :
        """
        Abstract class representing a set of conformers, each with a subset of rotamers

        Note: Abstract here mainly means that it has no pruning method. So, in practice, rotamers will not be identified
        """

        def __init__ (self) :
                """
                Creates an instance of the conformer class
                """
                Conformers.__init__(self)

                # Info that needs to be stored for every conformer
                self.conformer_data['rotamers'] = []

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

                duplicate = None
                if check_for_duplicates :
                        duplicate = self.find_duplicate(energy,coords)
                if duplicate is None :
                        self._add_if_not_duplicate(coords,energy)
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
                                        coords, energy = self._swap_conformers(duplicate, coords, energy)
                                self.rotamers[duplicate]._add_if_not_duplicate(coords,energy)
                        else :
                                self.copies[duplicate] += 1
                return duplicate

        def write (self, filename='base', write_rotamers=False, dirname='.', filetype='dcd') :
                """
                Write the conformers to file
                """
                if filetype == 'dcd' :
                        self.write_dcd(filename,write_rotamers,dirname=dirname)
                elif filetype == 'rkf' :
                        self.write_rkf(filename,write_rotamers,dirname=dirname)

        def write_dcd (self, filename='base', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in DCD format
                """
                Conformers.write_dcd(self,filename,dirname)
                if write_rotamers :
                        for i in range(len(self)) :
                                pathname = os.path.join(dirname,'rotamers_%s%i.dcd'%(filename,i))
                                dcd = DCDTrajectoryFile(pathname,mode='wb',ntap=len(self.molecule))
                                for i,crd in enumerate(self.rotamers[0].geometries) :
                                        dcd.write_next(coords=crd)
                                dcd.close()

        def write_rkf (self, filename='base', write_rotamers=False, dirname='.') :
                """
                Write the conformers to file in RKF format
                """
                Conformers.write_rkf(self,filename,dirname)
                if write_rotamers :
                        # The rotamer files can just be straightforward RKF trajectory files
                        for i in range(len(self)) :
                                pathname = os.path.join(dirname,'rotamers_%s%i.rkf'%(filename,i))
                                rkf = RKFTrajectoryFile(pathname,mode='wb',ntap=len(self.molecule))
                                for i,crd in enumerate(self.rotamers[0].geometries) :
                                        rkf.write_next(coords=crd)
                                rkf.close()

        def __str__ (self) :
                """
                Print conformer info
                """
                block = Conformers.__str__(self)
                lines = block.split('\n')
                newlines = [lines[0]+'%10s '%('#Rotamers')]
                for i,line in enumerate(lines[1:]) :
                        if len(line.split())==0 : continue
                        newline = line + '%10i'%(len(self.rotamers[i]))
                        newlines.append(newline)
                block = '\n'.join(newlines)
                return block

        # Private methods

        def _add_if_not_duplicate (self, coords, energy) :
                """
                Add the candidate (check for duplicate has already been done)
                """
                Conformers._add_if_not_duplicate(self,coords,energy)
                self.rotamers.append(self.__class__())
                self.rotamers[-1].prepare_state(self.molecule)

        def is_rotamer (self, iframe, coords) :
                """
                Checks if a certain coordinate/energy combination was already found
                """
                return False

        def _is_rmsd_smaller (self, coords, duplicate) :
                """
                Check if the new rotamer has a smaller RMSD to the lowest energy conformer than its conformer
                """
                rmsd, grad = compute_rmsd(self.geometries[duplicate],self.geometries[0],compute_grad=False)
                rmsd_new, grad = compute_rmsd(coords,self.geometries[0],compute_grad=False)
                return rmsd_new < rmsd

        def _is_not_rotamer_duplicate (self, coords, duplicate) :
                """
                Check if the new coordinates are not a duplicate of one of the rotamers
                """
                isrot = True
                for iframe in range(len(self.rotamers[duplicate])) :
                        if not self.rotamers[duplicate].is_rotamer(iframe, coords) :
                                isrot = False
                return isrot

