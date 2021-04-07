#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Abstract base class for the generation of conformers.
"""

import time
from scm.plams import Units
from scm.plams import Settings
from scm.plams import AMSJob
from ..optimizer import MoleculeOptimizer

__all__ = ['Generator']

class Generator :
        """
        Machine that extends a set of conformers
        """
        # https://xtb-docs.readthedocs.io/en/latest/optimization.html
        # The SCM units are Hartree/Angstrom. The CREST units may be Hartree/Bohr, but I am not sure. It would be weak settings.
        convergence_settings = {}
        convergence_settings['crude']     = {'Energy':1.e-4,'Gradients':1.e-2}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['sloppy']    = {'Energy':1.e-4,'Gradients':6.e-3}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['loose']     = {'Energy':5.e-5,'Gradients':4.e-3}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['lax']       = {'Energy':2.e-5,'Gradients':2.e-3}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['normal']    = {'Energy':5.e-6,'Gradients':1.e-3}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['tight']     = {'Energy':1.e-6,'Gradients':8.e-4}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['vtight']    = {'Energy':1.e-7,'Gradients':2.e-4}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['extreme']   = {'Energy':5.e-8,'Gradients':5.e-5}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['ams']       = {'Energy':5.e-5,'Gradients':5.e-3}# /Units.conversion_ratio('Bohr','Angstrom')}
        convergence_settings['ams_tight'] = {'Energy':5.e-7,'Gradients':5.e-5}# /Units.conversion_ratio('Bohr','Angstrom')}

        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Generator class

                * ``conformers`` -- A conformers object
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                self.name = 'base'

                # Molecule related settings (we assume the conformers object is ordered)
                self.conformers = conformers
                if conformers is None : 
                        raise Exception('A conformers object needs to be supplied')
                if self.conformers.molecule is None :
                        raise Exception('The conformers object must have a prepared state')
                self.mol = self.conformers.molecule

                # Energy related settings
                if engine_settings is None :
                        engine_settings = Settings()
                        engine_settings.DFTB.Model = 'GFN1-xTB'
                self.engine_settings = engine_settings
                self.nprocs_per_job = nprocs_per_job
                self.nprocs = nprocs
                self.optimizer = MoleculeOptimizer(self.mol,self.engine_settings,self.nprocs_per_job,nprocs)
                self.energy_threshold = energy_threshold
                self.min_energy = None

                # Timing settings
                self.steptime = None
                self.ngosteps = None

        def set_jobrunner (self, jobrunner) :
                """
                Set a jobrunner to organize parallelization

                * ``jobrunner`` -- Instance of the PLAMS Jobrunner class
                """
                self.optimizer.jobrunner = jobrunner
                self.nprocs = jobrunner.maxjobs

        def generate (self) :
                """
                Generate a conformer set, based on the CREST method
                """
                self.prepare_state()
                return self.conformers

        def prepare_state (self) :
                """
                Set the object variables
                """
                # Run a geometry optimization with convergence settings tight
                if len(self.conformers)==0 :
                        mol, energy = self._optimize_molecule()
                        self.min_energy = energy
                        self.mol = mol
                        self.conformers.add_conformer(self.mol.as_array(),energy)
                else : 
                        self.min_energy = self.conformers.energies[0]
                        self.mol.from_array(self.conformers.geometries[0])

        def optimize_and_filter (self, geometries, level='tight', name='go') :
                """
                Run the geometry optimizations for the results of the MTD runs and add to conformer set
                """
                # Tight geometry optimization and add to the instance conformer set
                convergence = self.convergence_settings[level]
                geometries, energies = self.optimizer.optimize_geometries(geometries, convergence, name=name)
                for crd,en in zip(geometries,energies) :
                        if crd is None : continue
                        if (en-self.conformers.energies[0]) < self.energy_threshold :
                                self.conformers.add_conformer(crd,en)

        def optimize_conformers (self, convergence_level, name='go') :
                """
                (Re)-Optimize the conformers currently in the set

                * ``convergence_level`` -- One of the convergence options ('tight', 'vtight', 'loose', etc')
                """
                self.conformers.optimize(convergence_level, self.optimizer, name=name)

        def _optimize_molecule (self) :
                """
                Optimize the instance variable `mol`
                """
                settings = self.optimizer.settings
                go_job = AMSJob(name='start',molecule=self.mol,settings=settings)
                go_job.settings.input.ams.GeometryOptimization['Convergence'] = self.convergence_settings['tight']
                # Maybe run this one in parallel (12 cores?)
                starttime = time.time()
                result = go_job.run()
                nsteps = result.readrkf('History','nEntries')
                energy = result.get_property_at_step(nsteps,'Energy') * Units.conversion_ratio('Hartree','kcal/mol')
                # Update the molecule, the minimum energy, and the conformer set
                mol = result.get_main_molecule()

                endtime = time.time()
                self.steptime = (endtime-starttime) / nsteps
                self.ngosteps = nsteps

                return mol, energy
