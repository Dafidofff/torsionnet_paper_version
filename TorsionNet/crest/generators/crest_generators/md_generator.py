#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Class for the generation of conformers from CREST molecular dynamics simulations
"""

import copy
import numpy
from scm.plams import Units
from scm.plams import Settings
from scm.plams import AMSJob
from scm.plams import JobRunner
from ..generator import Generator

__all__ = ['MDGenerator']

class MDGenerator (Generator) :
        """
        Machine that produces a set of conformers from molecular dynamics simulations
        """
        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                Generator.__init__(self, conformers, engine_settings, nprocs_per_job, energy_threshold, nprocs)

                # Settings for frame extraction
                self.blocksize = 1000 # Will be adjusted based on system size

                # MD settings
                temperature = 300.  # [400,500]
                self.mdsettings = Settings()
                self.mdsettings.TimeStep = 2.
                self.mdsettings.NSteps = None
                self.mdsettings.InitialVelocities.Temperature = temperature
                self.mdsettings.Thermostat.Temperature = temperature
                self.mdsettings.Thermostat.Type = 'Berendsen'
                self.mdsettings.Thermostat.Tau = 10. # Strong damping
                self.mdsettings.Trajectory.SamplingFreq = 1

                # Settings for the number of MD simulations
                self.temperatures = [400,500]
                self.ngeoms = 3     # According to the paper this is 3, but according to the executable it is 2
                self.multiples = 2  # According to the paper this is 2, but according to the executable it is 1
                self.nparams = [len(self.temperatures),self.ngeoms,self.multiples]
                self.settings_list = None

                # Parallelization settings
                self.jobrunner = JobRunner(parallel=True, maxjobs=nprocs)

        def set_jobrunner (self, jobrunner) :
                """
                Set a jobrunner to organize parallelization

                * ``jobrunner`` -- Instance of the PLAMS Jobrunner class
                """
                self.jobrunner = jobrunner
                self.optimizer.jobrunner = jobrunner
                self.nprocs = jobrunner.maxjobs

        def set_number_of_identical_mdruns (multiples) :
                """
                Sets the number of MD runs
                """
                self.multiples = multiples

        def set_number_of_starting_geometries (ngeoms) :
                """
                Set the number of conformers used to start the MD runs from
                """
                self.ngeoms = ngeoms

        def generate (self) :
                """
                Generate a conformer set, based on the CREST method
                """
                self.prepare_state()

                # Run the 12 MD jobs once
                results = self.run_jobs('md')

                # Extract a subset of frames for geometry optimization
                geometries, job_indices = self.extract_frames_from_mdresults(results)

                # Perform the geometry optimizations
                self.optimize_and_filter(geometries,name='md_go')

                # Update the molecule to the lowest energy conformer
                self.mol.from_array(self.conformers.geometries[0])
                if self.conformers.energies[0] < self.min_energy :
                        self.min_energy = self.conformers.energies[0]

                print ('Lowest MD energy: %20.5f'%(self.min_energy))

                return self.conformers

        def prepare_state (self) :
                """
                Prepare the settings for a metadynamics run

                See: DOI: 10.1039/c9cp06869d
                """
                # Turn all the hydrogen atoms to deuterium
                self._hydrogen_to_deuterium()

                # Run a geometry optimization with convergence settings tight
                Generator.prepare_state(self)

                # Determine the MD simulation time
                if self.mdsettings.NSteps is None :
                        self.mdsettings.NSteps = self._get_number_of_steps(factor=0.5)
                        self.blocksize = self._get_extraction_blocksize (self.mdsettings.NSteps)

                # Create the settings objects 
                self.settings_list = self._create_settings_list()

        def run_jobs (self, name='md', settings_list=None) :
                """
                Run the metadynamics jobs in parallel
                """
                result_list = []
                mdjobs = self._create_ams_jobs(name,settings_list)
                for job in mdjobs :
                        r = job.run(jobrunner=self.jobrunner)
                        result_list.append(r)
                return result_list

        def extract_frames_from_mdresults (self, results) :
                """
                Extract low energy frames
                """
                geometries = []
                job_indices = []
                for i,r in enumerate(results) :
                        # Read all energies and write them to file (the Gaussian energies should not be in here)
                        all_energies = r.get_history_property('PotentialEnergy','MDHistory')
        
                        # Extract the energies for the interesting frames
                        nblocks = int(numpy.ceil(len(all_energies) / self.blocksize))
                        tail = len(all_energies)%self.blocksize
                        energies = numpy.zeros(nblocks*self.blocksize)
                        energies[:len(all_energies)] = all_energies
                        energies = energies.reshape((nblocks,self.blocksize))
                        indices = energies.argmin(axis=1)
                        min_energies = energies[(range(nblocks),indices)]

                        # Adjust the last block for incompleteness
                        indices[-1] = energies[-1,:tail].argmin()
                        min_energies = energies[(range(nblocks),indices)]

                        indices = [(i_ind*self.blocksize)+ind for i_ind,ind in enumerate(indices)]

                        # Now read the corresponding coordinates (to Angstrom)
                        for ind in indices :
                                # Expensive step
                                crd = numpy.array(r.get_property_at_step(ind+1,'Coords','History'))
                                crd = crd.reshape((len(self.mol),3)) * Units.conversion_ratio('Bohr','Angstrom')
                                geometries.append(crd)
                                job_indices.append((i,ind))
                job_indices = numpy.array(job_indices)
                return geometries, job_indices

        ################
        # Private methods
        ################

        def _hydrogen_to_deuterium (self) :
                """
                Convert the H-atoms to D, for a smaller timestep
                """
                for at in self.mol.atoms :
                        if at.atnum == 1 :
                                at.properties.suffix = 'mass=2.014'

        def _get_number_of_steps (self, factor) :
                """
                Compute the number of steps
                """
                engine_settings = None
                if list(self.engine_settings.keys())[0] == 'DFTB' :
                        engine_settings = self.engine_settings
                mu = self.mol.compute_flexibility_factor(engine_settings,self.nprocs_per_job)
                mdtime, t_orig = self._metadynamics_time_from_flexibility(mu)
                #print ('Estimated time in ps: ',mdtime, t_orig)
                nsteps = int(mdtime * 1.e3 / self.mdsettings.TimeStep )
                # The MD time is half the MTD time
                nsteps = int(factor*nsteps)
                return nsteps

        def _get_extraction_blocksize (self, nsteps) :
                """
                Computes the blocksize for frame extraction from nsteps

                Note: Say we want 200 GOs, but blocksize should not be larger than 1000
                """
                blocksize = int((nsteps * numpy.prod(self.nparams)) / 200.)
                if blocksize > 1000 :
                        blocksize = 1000
                return blocksize

        def _metadynamics_time_from_flexibility (self, mu) :
                """
                Use the flexibility factor to compute the metadynamics time (in ps)
                """
                N_eff = len(self.mol) * mu
                t_orig = 0.1*(N_eff + 0.1*N_eff**2)
                t_mtd = t_orig
                if t_orig < 5:
                        t_mtd = 5.
                elif t_orig > 200:
                        t_mtd = 200.
                return t_mtd, t_orig

        def _create_settings_list (self) :
                """
                Create the list of settings objects
                """
                settings_list = []
                for i in range(self.ngeoms) :
                        # Later, three different geometries will be matched with these settings
                        for temperature in self.temperatures :
                                seeds = [iseed for iseed in range(self.multiples)]
                                if self.multiples == 1 :
                                        seeds = [None]
                                for seed in seeds :
                                        settings_list.append(self._create_mdsettings_object(temperature,seed))
                return settings_list

        def _create_mdsettings_object (self, temperature=300, seed=None) :
                """
                Create the full settings object from the temperature
                """
                s = Settings()
                s.runscript.nproc = self.nprocs_per_job
                engine_name = list(self.engine_settings.keys())[0]
                s.input[engine_name] = self.engine_settings[engine_name]
                s.input.ams.Task = 'MolecularDynamics'
                s.input.ams.MolecularDynamics = copy.deepcopy(self.mdsettings)

                s.input.ams.MolecularDynamics.InitialVelocities.Temperature = temperature
                s.input.ams.MolecularDynamics.Thermostat.Temperature = temperature

                if seed is not None :
                        s = self._set_random_velocities(s, temperature, seed)
                return s

        def _set_random_velocities (self, s, temperature, seed) :
                """
                Insert random velocities into settings object
                """
                if 'Temperature' in s.input.ams.MolecularDynamics.InitialVelocities :
                        del s.input.ams.MolecularDynamics.InitialVelocities.Temperature
                s.input.ams.MolecularDynamics.InitialVelocities.Type = 'Input'
                velocities = self._generate_random_velocities(temperature,seed)
                dic = {}
                #block = '\n' 
                for vec in velocities :
                        dic['%20.10f '%(vec[0])] = ' '.join(['%20.10f'%(v) for v in vec[1:]])
                        #block += ' '.join(['%20.10f'%(v) for v in vec]) + '\n'
                #block = block[:-1]
                #s.input.ams.MolecularDynamics.InitialVelocities.Values = block
                s.input.ams.MolecularDynamics.InitialVelocities.Values = dic
                return s

        def _generate_random_velocities (self, temperature, seed=None) :
                """
                Generate random velocities

                * ``temperature`` -- Temperature in K
                """ 
                kB = Units.constants['k_B']
                nats = len(self.mol)
                # masses in kg
                masses = numpy.array([at.mass for at in self.mol.atoms]) * 1.e-2 / Units.constants['Avogadro_constant']
                random_number_generator = numpy.random.mtrand.RandomState()
                if seed is not None :
                        random_number_generator.seed(seed)
                xi = random_number_generator.standard_normal((len(masses),3))
                # Generate velocities in m/s
                velocities = xi * numpy.sqrt(temperature * kB / masses)[:,numpy.newaxis]
                # Now convert to Angstrom/fs
                velocities *= 1e10 / 1e15
                return velocities

        def _create_ams_jobs (self, name, settings_list=None) :
                """
                Creat the 12 AMS jobs from the settings
                """
                if settings_list is None :
                        settings_list = self.settings_list

                # If we do not have many conformers, we do not have to run a lot of MD
                if len(self.conformers) < self.ngeoms :
                        counter = 0
                        indices = []
                        for ig in range(self.ngeoms) :
                                for t in self.temperatures :
                                        for m in range(self.multiples) :
                                                if not ig >= len(self.conformers) :
                                                        indices.append(counter)
                                                counter += 1
                        settings_list = [settings_list[i] for i in indices]

                njobs = len(settings_list)
                nsettings_per_geom = int(njobs/self.ngeoms) # 6
                mdjobs = [AMSJob(name='%s%i'%(name,i),molecule=self.mol,settings=s) for i,s in enumerate(settings_list)]
                for i,crd in enumerate(self.conformers.geometries[:self.ngeoms]) :
                        start = i*nsettings_per_geom
                        end = (i+1)*nsettings_per_geom
                        for job in mdjobs[start:end] :
                                job.molecule.from_array(crd)
                return mdjobs
