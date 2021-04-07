#!/usr/bin/env python

import os
from scm.plams import AMSJob
from scm.plams import Settings
from scm.plams import Units
from scm.plams import TrajectoryFile
from scm.plams import FileError
from scm.plams import JobRunner

__all__ = ['MoleculeOptimizer']

class MoleculeOptimizer :
        """
        Machine that can optimize the geometry of a set of structures
        """
        def __init__ (self, mol, engine_settings=None, nprocs_per_job=1, nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                * ``nprocs_per_job``  -- Number of processors used for each single call to AMS
                * ``nprocs``          -- Maximum number of parallel AMS processes
                """
                if engine_settings is None :
                        engine_settings = Settings()
                        engine_settings.DFTB.Model = 'GFN1-xTB'

                self.mol = mol
                self.settings = Settings()
                self.settings.runscript.nproc = nprocs_per_job
                self.settings.input.ams.Task = 'GeometryOptimization'
                engine_name = list(engine_settings.keys())[0]
                self.settings.input[engine_name] = engine_settings[engine_name]

                # Wether or not to keep the files after running
                self.keep = None     # 'all' is the PLAMS default
                self.jobrunner = JobRunner(parallel=True, maxjobs=nprocs)

        def set_jobrunner (self, jobrunner) :
                """
                Set a jobrunner to organize parallelization

                * ``jobrunner`` -- Instance of the PLAMS Jobrunner class
                """
                self.jobrunner = jobrunner

        def set_cleaning_preference (self, keep=None) :
                """
                * ``keep`` -- None or 'all'. 
                              The former will clean PLAMS folders of all files after runtime,
                              the latter will leave all files as they are.
                              See PLAMS |cleaning| for more details.
                """
                self.keep = keep

        def optimize_geometries (self, geometries, convergence={'Energy':1.e-5,'Gradients':1.e-3}, name='go') :
                """
                Optimize a set of geometries

                @param geometries: An iterator over molecular coordinates (matching self.mol)
                """
                # Set the GO convergence
                self.settings.input.ams.GeometryOptimization['Convergence'] = convergence

                resultlist = []
                for i,data in enumerate(geometries) :
                        if isinstance(geometries, TrajectoryFile) :
                                # This returns both coordinates and cell info
                                coords = data[0]
                        else : 
                                # Inthis case geometries is most likely a simple list of coordinates
                                coords = data
                        self.mol.from_array(coords)
                        job = AMSJob(name='%s%i'%(name,i), molecule=self.mol, settings=self.settings)
                        resultlist.append(job.run(jobrunner=self.jobrunner))

                optimized_geometries = []
                energies = []
                for i,r in enumerate(resultlist) :
                        try :
                                # I do it like this to make sure we hit the guardian, I think
                                nsteps = r.readrkf('History','nEntries')
                        except FileError :
                                pass
                        # If the geometry was copied from another directory, then it is not new so we do not need it.
                        if r.job.status == 'failed' or not os.path.isfile(os.path.join(r.job.path,'ams.rkf')) :
                                optimized_geometries.append(None)
                                energies.append(None)
                                continue
                        energy = r.get_property_at_step(nsteps,'Energy') * Units.conversion_ratio('Hartree','kcal/mol')
                        #energy = r.get_energy() * Units.conversion_ratio('Hartree','kcal/mol')
                        coords = r.get_main_molecule().as_array()
                        optimized_geometries.append(coords)
                        energies.append(energy)

                # Remove files
                for i,r in enumerate(resultlist) :
                        r._clean(self.keep)

                return optimized_geometries, energies
                        
                        
