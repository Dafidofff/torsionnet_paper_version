#!/usr/bin/env python

"""
Author: Rosa Bulo, 2021

Class for the generation of conformers using RDKit
"""

from scm.plams import get_conformations
from ..generator import Generator

__all__ = ['RDKitGenerator']

class RDKitGenerator (Generator):
        """
        Machine that generates a set of unique conformers using RDKit

        DOI: 10.1021/acs.jcim.5b00654
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

                self.name = 'rdkit'

                # RDKit procedure settings
                self.ngeoms = None
                self.factor = 2    # Should be 370 based on my experience

        def set_number_initial_conformers (self, ngeoms=None, min_confs=10, max_confs=1000) :
                """
                Set the number of conformers created by RDKit, before GO and filtering
                """
                self.ngeoms = ngeoms
                if ngeoms is None :
                        self.ngeoms = self._estimate_nconformers(min_confs,max_confs)

        def generate (self) :
                """
                Generate a conformer set, using RDKit
                """
                self.prepare_state()

                # Set the number of initially generated conformers
                if self.ngeoms is None :
                        self.set_number_initial_conformers()
                print ('The number of conformers initially generated will be %s'%(self.ngeoms))

                # Generate the geometries
                geometries = self.generate_geometries()

                # Optimize the geometries
                self.optimize_and_filter(geometries,level='ams_tight')

                # Update the molecule to the lowest energy conformer
                self.mol.from_array(self.conformers.geometries[0])
                if self.conformers.energies[0] < self.min_energy :
                        self.min_energy = self.conformers.energies[0]

                return self.conformers

        def generate_geometries (self) :
                """
                Call RDKit to generate a set of conformers
                """
                conformers = get_conformations(self.mol, self.ngeoms, enforceChirality=True)
                geometries = [plmol.as_array() for plmol in conformers]

                return geometries

        def _estimate_nconformers (self, min_confs=10, max_confs=1000) :
                """
                Estimate the best number of conformers, based on number of rotational bonds

                * ``min_confs`` -- The minimum number of conformers
                * ``max_confs`` -- The maximum number of conformers
                """
                nconformers = int(self.mol.predict_nconformers()) * self.factor
                if nconformers < min_confs :
                        nconformers = min_confs
                elif nconformers > max_confs :
                        nconformers = max_confs
                return nconformers


