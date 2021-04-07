#!/usr/bin/env python

"""
Author: Ravi Ramautar, 2020

Class for the generation of conformers using the CREST workflow
"""

import sys
import os
from ..generator import Generator
from .md_generator import MDGenerator
from .metadynamics_generator import MetadynamicsGenerator
from .gc_generator import GCGenerator

__all__ = ['CRESTGenerator']

class CRESTGenerator (Generator) :
        """
        Machine that produces a set of conformers using the CREST workflow
        """
        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                Generator.__init__(self, conformers, engine_settings, nprocs_per_job, energy_threshold, nprocs)

                self.name = 'crest'

                # Set up the individual generators
                self.mtd_generator = MetadynamicsGenerator(self.conformers,self.engine_settings,nprocs_per_job,energy_threshold,nprocs)
                self.md_generator  =           MDGenerator(self.conformers,self.engine_settings,nprocs_per_job,energy_threshold,nprocs)
                self.gc_generator  =           GCGenerator(self.conformers,self.engine_settings,nprocs_per_job,energy_threshold,nprocs)

                # Parameters for the procedure
                self.gc_step = True

        def generate (self) :
                """
                Generate a conformer set, based on the CREST method
                """
                self.prepare_state()
                
                dirname = 'conf_tmpdir'
                if not os.path.isdir(dirname) :
                        os.mkdir(dirname)

                run = True
                counter = 0
                while run :
                        print ('=========================================')
                        print ('=     CREST Generation Cycle %2i         ='%(counter))
                        print ('=========================================\n')
                        print ('Starting MTD cycle')
                        print ('------------------')
                        sys.stdout.flush()
                        self.mtd_generator.generate()
                        self.conformers.write('crest_mtd',dirname=dirname)
                        print ('MTD cycle done\n')
                        print ('Starting MD cycle')
                        print ('-----------------')
                        sys.stdout.flush()
                        self.md_generator.generate()
                        self.conformers.write('crest_md',dirname=dirname)
                        print ('MD cycle done\n')
                        energy = self.md_generator.min_energy
                        if self.md_generator.min_energy < self.min_energy :
                                self.min_energy = self.md_generator.min_energy
                        else :
                                run = False
                        if not run and self.gc_step :
                                # If the md_generator found a lower energy, go immediately to restart
                                print ('Starting GC cycle')
                                print ('-----------------')
                                sys.stdout.flush()
                                self.gc_generator.generate()
                                self.conformers.write('crest_gc',dirname=dirname)
                                print ('GC cycle done\n')
                                energy = self.gc_generator.min_energy
                                if self.gc_generator.min_energy < self.min_energy :
                                        self.min_energy = self.gc_generator.min_energy
                                        run = True
                        print ('Lowest energy CREST cycle %2i: %20.5f\n'%(counter,self.min_energy))
                        counter += 1
                        if counter > 10 :
                                # Just for now, to avoid an infinite loop!
                                break

                print ('Number of conformers: %8i\n'%(len(self.conformers)))

                # Now do a vtight geometry optimization on top
                print ('Final geometry optimizations')
                print ('----------------------------')
                sys.stdout.flush()
                self.optimize_conformers('vtight',name='final_go')
                print ('Number of conformers: %8i\n'%(len(self.conformers)))

                return self.conformers

        def prepare_state (self) :
                """     
                Set the object variables
                """
                Generator.prepare_state(self)
                self.mtd_generator.steptime = self.steptime
                self.mtd_generator.ngosteps = self.ngosteps
                self.mtd_generator.prepare_state()

                self.min_energy = self.mtd_generator.min_energy
