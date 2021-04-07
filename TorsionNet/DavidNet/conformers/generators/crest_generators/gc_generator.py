#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Class for the generation of conformers from CREST genetic structure crossing.
This is not a standalone conformer generator, but extends an existing conformer set.
"""

import numpy
from multiprocessing import Pool
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
                self.parallel = True  # Wether the generation of the geometries occurs in parallel or not

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
                max_geoms = (len(self.conformers)*(len(self.conformers)-1))
                print ('Creating %s geometries.... '%(max_geoms))

                if not self.parallel :
                        # Serial generation of combinatorial geometries
                        geometries,rmsds = self.get_geometries()
                else :
                        # Parallel generation of combinatorial geometries
                        self_trimmed = TrimmedGCGenerator()
                        self_trimmed.prepare_state(self)
                        nprocs = self.nprocs*self.nprocs_per_job
                        blocksize = numpy.ceil(len(self.conformers) / nprocs)
                        pool = Pool(processes=nprocs)
                        processlist = []
                        for iproc in range(nprocs) :
                                indices = [i for i in range(len(self.conformers)) if i >= iproc*blocksize and i < (iproc+1)*blocksize]
                                result = pool.apply_async(TrimmedGCGenerator.get_geometries, (self_trimmed,indices,self.conformers.geometries))
                                processlist.append(result)
        
                        # Close the parallel processes
                        pool.close()
                        pool.join()
                        pool.terminate()
        
                        # Get the results from the parallel processes
                        geometries = []
                        rmsds = []
                        for result in processlist :
                                geom, rmsd = result.get()
                                geometries += geom
                                rmsds += rmsd

                # Select the ones we want to keep
                rmsds = numpy.array(rmsds)

                if max_geoms == 0 : max_geoms = 1
                print ('Fraction accepted geometries: ',len(rmsds)/max_geoms, len(rmsds))

                # Select the ngeoms geometries with the highest RMSD from the reference
                indices = rmsds.argsort()[::-1][:self.ngeoms]
                geometries = [geometries[ind] for ind in indices]

                return geometries

        def get_geometries (self, indices=None, geometries=None) :
                """
                Create new geometries, by combining differences

                Note: No more than self.ngeoms new geometries are created

                * ``indices``    -- Indices of the geometries over which the second loop iterates
                * ``geometries`` -- All the geometries in the conformer set
                """
                if indices is None :
                        indices = [i for i in range(len(self.conformers))]
                if geometries is None :
                        geometries = self.conformers.geometries

                #print ('NGeoms process: ',len(indices)*len(geometries))

                # Generate the combinatorial geometries (maybe this loop should be truncated?)
                # I could use the clustering feature for that (select lowest energy structure from each cluster)
                crds_ref = geometries[0]
                new_geometries = []
                rmsds = []
                max_rmsds = numpy.zeros(self.ngeoms)
                zmat_ref = self.zmatrix.get_values(crds_ref)
                for i,crds_i in enumerate(geometries) :
                        if i == 0 : continue
                        zmat_i = self.zmatrix.get_values(crds_i)
                        # Do we loop over all geometries here? Or just j > i?
                        for j in indices :
                                crds_j = geometries[j]
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
                                rmsd, grad = compute_rmsd(crds_ref,crd,compute_grad=False)
                                if len(new_geometries) >= self.ngeoms and rmsd <= max_rmsds.min() :
                                        continue
                                new_geometries.append(crd)
                                rmsds.append(rmsd)
                                max_rmsds[max_rmsds.argmin()] = rmsd

                return new_geometries, rmsds

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

class TrimmedGCGenerator :
        """
        Trimmed version of the GCGenerator class that can easily be pickled.
        Does not have the geometry optimization functionality
        """
        def prepare_state (self, gcgenerator) :
                """
                Prepatres the trimmed version of the GCGenerator object

                * ``gcgenerator`` -- Instance of the big GCGenerator class
                """
                self.zmatrix = gcgenerator.zmatrix
                self.ngeoms = gcgenerator.ngeoms

                # Settings for computation of the coordination number
                self.ref_coordnums = gcgenerator.ref_coordnums
                self.clash_theshold = gcgenerator.clash_theshold
                self.power = gcgenerator.power
                self.radius_matrix = gcgenerator.radius_matrix
                
        def _compute_coordination_numbers (self, coords) :
                """
                Checks a geometry for too close atomic contacts
        
                Note: The Grimme code uses a D3 coordination number here
                """
                coordnums = GCGenerator._compute_coordination_numbers(self, coords)
                return coordnums

        def get_geometries (self, indices, geometries) :
                """
                Use the CREST GC algorithm to create additional geometries, by matching all coordinates to the subset `indices`.
        
                * ``indices``    -- List of geometry indices pointing to a subset of geometries in the conformer set
                * ``geometries`` -- All the geometries in the conformers object of the big GCGenerator object
                """
                new_geometries, rmsds = GCGenerator.get_geometries(self, indices, geometries)
                return new_geometries, rmsds
                
