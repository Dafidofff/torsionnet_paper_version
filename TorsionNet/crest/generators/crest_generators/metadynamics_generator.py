#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Class for the generation of conformers from CREST metadynamics simulations
"""

import os
import copy
import numpy
from scm.plams import KFFile
from scm.plams import Settings
from scm.plams import AMSJob
from scm.plams import AMSResults
from .md_generator import MDGenerator
from ..generator import Generator

__all__ = ['MetadynamicsGenerator']

class MetadynamicsGenerator (MDGenerator) :
        """
        Machine that produces a set of conformers from CREST metadynamiss simulations
        """
        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``conformers`` -- A conformers object
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                MDGenerator.__init__(self, conformers, engine_settings, nprocs_per_job, energy_threshold, nprocs)

                # Input info
                self.input_coords = None

                # Metadynamics settings
                self.nparams = [3,4] # Number of widths and heights
                self.mtdsettings = Settings()
                self.mtdsettings.Width = None
                self.mtdsettings.Height = None
                self.mtdsettings.NSteps = int(1000 / self.mdsettings.TimeStep)
                self.mtdsettings.NGaussiansMax = 10
                #self.mtdsettings.AddEnergy = False
                
        def generate (self) :
                """
                Generate a conformer set, based on the CREST method
                """
                if self.min_energy != self.conformers.energies[0] :
                #if self.min_energy is None :
                        self.prepare_state()
                print ('Starting energy:              %20.5f'%(self.min_energy))

                # Run a maximum of 5 MTD cycles
                for i in range(5) :
                        print ('Running MTD Step %2i'%(i))
                        again = self.runMTDcycle()
                        print ('MTD Step %2i done'%(i))
                        print ('Lowest energy:                %20.5f'%(self.min_energy))
                        if not again and i>0 :
                                break
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

                # Determine the MTD simulation time
                if self.mdsettings.NSteps is None :
                        self.mdsettings.NSteps = self._get_number_of_steps(factor=1.)
                        self.blocksize = self._get_extraction_blocksize (self.mdsettings.NSteps)

                # Collect the widths and heights of Gaussians in the MTD simulations and create the settings objects
                self.settings_list = self._create_settings_list()

                # Estimate simulation time
                simtime = self.mdsettings.NSteps * self.mdsettings.TimeStep / 1.e3
                print ('MTD simulation time in ps: ', simtime)
                timestring = self._estimate_walltime_mtdstep()
                print ('Estimated time for a metadynamics step: %s'%(timestring))

        def runMTDcycle (self, MTDdirname=None) :
                """
                Runs a single cycle of 12 MTD jobs, and updates the conformers object

                * ``MTDdirname`` -- A PLAMS directory containing previously computed MTD results

                Returns a logical stating wether or not another cycle is desirable
                """
                # Should another cycle be run after this one?
                again = False

                # Produce the MTD trajectories
                if MTDdirname is None :
                        settings_list = None
                        # If the molecule did not change, then we have to change something else (velocities)
                        if self.input_coords is not None :
                                if (self.input_coords == self.mol.as_array()).all() :
                                        settings_list = self._get_updated_settings_list()
                        results = self.run_jobs('mtd',settings_list)
                else :
                        results = self.load_MTDjob_results(MTDdirname)

                # Extract a subset of frames for geometry optimization
                geometries, job_indices = self.extract_frames_from_mdresults(results)

                # Perform the two sets of geometry optimizations and update the conformers object
                best_job = self.optimize_and_filter(geometries, job_indices)

                # Update the molecule to the lowest energy conformer
                self.mol.from_array(self.conformers.geometries[0])
                if self.conformers.energies[0] < self.min_energy :
                        again = True
                        self.min_energy = self.conformers.energies[0]

                # Possibly update the settings in the jobs, to correspond to the best settings

                return again

        def load_MTDjob_results (self, dirname) :
                """
                Load MTD results from dirname
                """
                # Update self.mol?
       
                # Collect the MTD path and input names 
                dirnames = os.listdir(dirname)
                mtd_dirnames = [dn for dn in dirnames if 'mtd' in dn]
                # The directories need to be sorted
                nums = numpy.array([int(dn[3:]) for dn in mtd_dirnames]).argsort()
                mtd_dirnames = [mtd_dirnames[ind] for ind in nums]

                # I want the rkffile and the infile
                pathnames = [os.path.join(dirname,dn) for dn in mtd_dirnames]
                rkffilenames = [os.path.join(pn,'ams.rkf') for pn in pathnames]
                infilenames = [os.path.join(pn,dn+'.in') for pn,dn in zip(pathnames,mtd_dirnames)]

                if len(rkffilenames) != len(self.settings_list) :
                        raise Exception('Wrong number of MTD jobs in directory')

                # Use the infile to check if these results match the preprepared jobs
                mtdjobs = self._create_ams_jobs ('mtd')
                for i,fn in enumerate(infilenames ) :
                        infile = open(fn)
                        text = infile.read()
                        infile.close()
                        if text != mtdjobs[i].get_input() :
                                raise Exception ('Input of job %i does not match file found'%(i))

                
                results_list = len(rkffilenames) * [1]
                for i,job in enumerate(mtdjobs) :
                        job.path = os.path.dirname(rkffilenames[i])
                        job.status = 'successful'
                        results_list[i] = AMSResults(job)
                        results_list[i].files = [rkffilenames[i]]
                        results_list[i].rkfs['ams'] = KFFile(rkffilenames[i])
                        mol = results_list[i].get_main_molecule()
                return results_list

        def optimize_and_filter (self, geometries, job_indices) :
                """
                Run the geometry optimizations for the results of the MTD runs
                """
                # Crude (I am not completely sure that the CREST gradients are in Hartree/Bohr or Hartree/ANgstrom)
                convergence = self.convergence_settings['crude']
                geometries, energies = self.optimizer.optimize_geometries(geometries, convergence, name='mtdcrude_go')
                remaining_energies = numpy.array([en for en in energies if not en is None])
                index_min = numpy.array(remaining_energies).argmin()
                best_job = job_indices[index_min][0]
                
                # This has to be a new conformer set of the same type as the instance conformer set
                local_conformers = self.conformers.copy()
                local_conformers.clear()
                min_energy = remaining_energies.min()
                for crd,en in zip(geometries,energies) :
                        if crd is None : continue
                        if (en-min_energy) < self.energy_threshold*2 :
                                local_conformers.add_conformer(crd,en)
                # Create a list of geometries includig all conformers and rotamers
                geometries = local_conformers.geometries
                geometries += [crds for rotamers in local_conformers.rotamers for crds in rotamers.geometries]

                MDGenerator.optimize_and_filter(self,geometries,name='mtdtight_go')
                return best_job

        ################
        # Private methods
        ################

        def get_Gaussian_widths (self, nvalues=3) :
                """
                Returns a list of Gaussian widths
                """
                # Alphas are the widths defined in Bohr-2 (G(s) = k*exp(-alpha*(s-s0)^2)), between 0.1 and 1.3
                min_a = 0.1
                max_a = 1.3
                alphas = ((numpy.arange(nvalues)/(nvalues-1)) * (max_a-min_a)) + min_a
                f = lambda a: numpy.sqrt(1/(2*a))
                widths = [f(a) for a in alphas]
                return widths

        def get_Gaussian_heights (self, nvalues=4) :
                """
                Returns a list of Gaussian heights
                """
                # The Gaussian heights are in Hartree: h/N = (0.75e-3,3.e-3)
                nats = len(self.mol)
                min_h = 0.75e-3 * nats
                max_h = 3.e-3 * nats
                heights = ((numpy.arange(nvalues)/(nvalues-1)) * (max_h-min_h)) + min_h
                return heights

        def _create_settings_list (self) :
                """
                Create the list of settings objects
                """
                widths = self.get_Gaussian_widths(self.nparams[0])
                heights = self.get_Gaussian_heights(self.nparams[1])
                settings_list = []
                for w in widths :
                        for h in heights :
                                settings_list.append(self._create_mdsettings_object(w,h))
                return settings_list

        def _get_updated_settings_list (self) :
                """
                Update the settings list with different starting velocities
                """
                widths = list(set([s.input.ams.MolecularDynamics.CRESTMTD.Width for s in self.settings_list]))
                heights = list(set([s.input.ams.MolecularDynamics.CRESTMTD.Height for s in self.settings_list]))
                seeds = numpy.random.permutation(numpy.arange(len(self.settings_list))+1).reshape((len(widths),len(heights)))

                settings_list = []
                for i,w in enumerate(widths) :
                        for j,h in enumerate(heights) :
                                settings_list.append(self._create_mdsettings_object(w,h,seed=seeds[i,j]))
                return settings_list

        def _create_mdsettings_object (self, width, height, temperature=300, seed=None) :
                """
                Returns an AMS settings object for a CREST MTD run
                """
                # General settings
                s = MDGenerator._create_mdsettings_object(self, temperature, seed)

                # Set the MTD settings
                s.input.ams.MolecularDynamics.CRESTMTD = copy.deepcopy(self.mtdsettings)
                s.input.ams.MolecularDynamics.CRESTMTD.Width = width
                s.input.ams.MolecularDynamics.CRESTMTD.Height = height
        
                # Ask for the engine info to be printed
                #s.input.ams.EngineDebugging.NeverQuiet = True

                return s
        
        def _create_ams_jobs (self, name, settings_list=None) :
                """
                Create the 12 AMS jobs
                """
                if settings_list is None :
                        settings_list = self.settings_list
                self.input_coords = self.mol.as_array()
                amsjobs = [AMSJob(name='%s%i'%(name,i),molecule=self.mol,settings=s) for i,s in enumerate(settings_list)]
                return amsjobs

        def _estimate_walltime_mtdstep (self) :
                """
                Use the walltime from a single GO to estimate the walltime of an MTD step (including GOs)
                """
                mtd_walltime = self.mdsettings.NSteps * self.steptime * int(numpy.prod(self.nparams)/self.nprocs)
                ngos = int(self.mdsettings.NSteps * numpy.prod(self.nparams) / self.blocksize)
                go_walltime = 2 * self.steptime * self.ngosteps * ngos / self.nprocs
                tottime = mtd_walltime + go_walltime
                hours = int(tottime/3600)
                minutes = int((tottime/60))%60
                seconds = tottime%60
                return '%02ih:%02im:%02is'%(hours,minutes,seconds)

