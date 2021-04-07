#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Class for the generation of conformers from CREST genetic structure crossing.
This is not a standalone conformer generator, but extends an existing conformer set.
"""

import numpy
from scm.plams import distance_array
from scm.plams import PeriodicTable
from scm.flexmd.structuraldescriptors.rmsd import compute_rmsd
from ...molecularsystem.zmatrix import ZMatrix
from ..generator import Generator

__all__ = ['GCGenerator']

class GCGenerator (Generator):
        """
        Machine that extends a set of conformers using genetic structure crossing

        DOI: 10.1002/anie.201708266 (Supporting Information)
        """
        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``conformers`` -- A conformers object
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                Generator.__init__(self, conformers, engine_settings, nprocs_per_job, energy_threshold, nprocs)

                # Molecule related settings (we assume the conformers object is ordered)
                self.zmatrix = ZMatrix()
                self.zmatrix.prepare_state(self.mol)
                self.radius_matrix = None      # parameter for the coordination number
                self.power         = 6     # parameter for the coordinattionm number
                self.clash_theshold = 0.5  # Maximum allowed increase in coordination number per atom

                # GC procedure settings
                self.ngeoms = 2000

        def generate (self) :
                """
                Generate a conformer set, based on the CREST method
                """
                self.prepare_state()

                # Generate the combinatorial geometries
                geometries = []
                # The zmatrix is only created if the molecule has a backbone of at least 4 atoms
                if self.zmatrix.atom_list is not None :
                        geometries = self.create_combinatorial_geometries()

                # Optimize the geometries
                self.optimize_and_filter(geometries,name='gc_go')

                # Update the molecule to the lowest energy conformer
                self.mol.from_array(self.conformers.geometries[0])
                if self.conformers.energies[0] < self.min_energy :
                        self.min_energy = self.conformers.energies[0]

                return self.conformers

        def prepare_state (self) :
                """
                Set the object variables
                """
                Generator.prepare_state(self)

                # Set the matrix (?) of atomic radii
                nats = len(self.mol)
                radii = numpy.array([PeriodicTable.get_radius(at.symbol) for at in self.mol.atoms])
                ones = numpy.ones((nats,nats))
                self.radius_matrix = ((ones*radii.reshape((1,nats))) + (ones*radii.reshape((nats,1)))) #/ 2
                self.ref_coordnums = self._compute_coordination_numbers(self.mol.as_array())

        def create_combinatorial_geometries (self) :
                """
                Create new geometries, by combining differences

                Note: No more than self.ngeoms new geometries are created

                FIXME: The conversions to and from zmatrix (especially the latter) are the really slow steps
                """
                from scm.plams import create_xyz_string

                max_geoms = (len(self.conformers)*(len(self.conformers)-1))
                print ('Creating %s geometries.... '%(max_geoms))

                # Generate the combinatorial geometries (maybe this loop should be truncated?)
                # I could use the clustering feature for that (select lowest energy structure from each cluster)
                # But first: Parallelize it! (per process I can store only the 2000 highest RMSDs, like now)
                geometries = []
                rmsds = []
                max_rmsds = numpy.zeros(self.ngeoms)
                crds_ref = self.conformers.geometries[0]
                zmat_ref = self.zmatrix.get_values(crds_ref)
                for i,crds_i in enumerate(self.conformers.geometries) :
                        if i == 0 : continue
                        zmat_i = self.zmatrix.get_values(crds_i)
                        # Do we loop over all geometries here? Or just j > i?
                        for j,crds_j in enumerate(self.conformers.geometries) :
                                if i==j : continue
                                zmat_j = self.zmatrix.get_values(crds_j)
                                zmat = zmat_ref + (zmat_j-zmat_i)
                                # Convert back to cartesian
                                crd = self.zmatrix.get_cartesian_coords(zmat)

                                # Only store is there are no clashes
                                change = self._compute_coordination_numbers(crd) - self.ref_coordnums
                                if change.max() > self.clash_theshold :
                                        #print ('Bad structure found: ',i,j)
                                        continue
                                # Only store if RMSD is large enough
                                rmsd, grad = compute_rmsd(self.conformers.geometries[0],crd,compute_grad=False)
                                if len(geometries) >= self.ngeoms and rmsd <= max_rmsds.min() :
                                        continue
                                geometries.append(crd)
                                rmsds.append(rmsd)
                                max_rmsds[max_rmsds.argmin()] = rmsd
                rmsds = numpy.array(rmsds)

                if max_geoms == 0 : max_geoms = 1
                print ('Fraction accepted geometries: ',len(rmsds)/max_geoms, len(rmsds))

                # Select the ngeoms geometries with the highest RMSD from the reference
                indices = rmsds.argsort()[::-1][:self.ngeoms]
                geometries = [geometries[ind] for ind in indices]

                return geometries

        def _compute_coordination_numbers (self, coords) :
                """
                Checks a geometry for too close atomic contacts

                Note: The Grimme code uses a D3 coordination number here
                """
                nats = len(coords)
                n = self.power
                m = self.power*2
                
                # Compute a distance matrix
                # FIXME: Use neighborlists here somehow?
                dist_matrix = distance_array(coords,coords)

                # Compute coordination numbers (need to get the zero out first)
                frac_mat = dist_matrix / self.radius_matrix
                upper_frac = frac_mat[numpy.triu_indices(nats,k=1)]
                lower_frac = frac_mat[numpy.tril_indices(nats,k=-1)]
                upper_coordnums = ((1-upper_frac**n) / (1-upper_frac**m))
                lower_coordnums = ((1-lower_frac**n) / (1-lower_frac**m))
                coordnums = numpy.zeros((nats,nats))
                coordnums[numpy.triu_indices(nats,k=1)] = upper_coordnums
                coordnums[numpy.tril_indices(nats,k=-1)] = lower_coordnums
                coordnums = coordnums.sum(axis=1)

                return coordnums
