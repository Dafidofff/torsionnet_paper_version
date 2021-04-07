#!/usr/bin/env python

"""
Author: Rosa Bulo, 2021

Class for the generation of conformers by enumerating over torsion angles
"""
import copy
import numpy
from scm.plams import get_conformations
from scm.flexmd import locate_rings
from scm.flexmd import pdb_from_plamsmol
from scm.flexmd import MDMolecule
from ..generator import Generator

__all__ = ['TorsionGenerator']

class TorsionGenerator (Generator):
        """
        Machine that generates conformers by enumerating over rotatable torsion angles
        """
        def __init__ (self, conformers, engine_settings=None, nprocs_per_job=1, energy_threshold=6., nprocs=1) :
                """
                Initiates an instance of the Optimizer class

                * ``conformers`` -- A conformers object
                * ``engine_settings`` -- PLAMS Settings object:
                                         engine_settings = Settings()
                                         engine_settings.DFTB.Model = 'GFN1-xTB'
                """
                Generator.__init__(self, conformers, engine_settings, nprocs_per_job, energy_threshold)

                self.name = 'torsion'

                # Molecule related variables
                self.pdb = None
                self.rings = None
                self.torsion_atoms = None

                # Enumeration procedure settings
                self.dtheta = 60.
                self.threshold = 1.1 # For the rings

        def prepare_state (self) :
                """
                Set the object variables
                """
                Generator.prepare_state(self)

                # Prepare the rings
                self.pdb = pdb_from_plamsmol(self.mol)
                mdmol = MDMolecule(pdb=self.pdb)
                self.rings = numpy.array(locate_rings(mdmol,[i for i in range(len(self.mol))]))

                # Define the torsion angles
                self.torsion_atoms = sorted(self.mol.get_rotatable_torsions(include_rings=False))

        def set_angle (self, angle) :
                """
                Set the angle over which to rotate the torsions
                """
                if 360%angle != 0 :
                        raise Exception('Angle has to be a fraction of 360 degrees')

                self.dtheta = angle

        def generate (self) :
                """
                Generate a conformer set, using RDKit
                """
                self.prepare_state()

                # Generate the geometries
                geometries = self.generate_geometries()
                print ('%s initial geometries'%(len(geometries)))

                # Optimize the geometries
                self.optimize_and_filter(geometries,level='ams_tight')

                # Update the molecule to the lowest energy conformer
                self.mol.from_array(self.conformers.geometries[0])
                if self.conformers.energies[0] < self.min_energy :
                        self.min_energy = self.conformers.energies[0]

                return self.conformers

        def generate_geometries (self) :
                """
                Generate a set of conformers by rotating around the rotatable bonds
                """
                nrotations = int(360/self.dtheta)

                nats = len(self.mol)
                self.pdb.coords = self.mol.as_array()
                
                # Create all the new geometries
                geometries = [numpy.array(self.pdb.coords)]
                for i,tor in enumerate(self.torsion_atoms) :
                        print (i, len(geometries))
                        new_geometries = []
                        for crd in geometries :
                                self.pdb.coords = copy.deepcopy(crd)
                                for irot in range(nrotations-1) :
                                        # Rotate full circle around torsions by ntheta degrees
                                        self.pdb.change_torsion(tor,self.dtheta)
                                        new_crd = copy.deepcopy(self.pdb.coords)
                                        new_geometries.append(new_crd)
                        geometries += new_geometries

                return geometries

        def generate_ring_geometries (self, ring_index=0) :
                """
                Generate the geometries using the torsions in the rings
                """
                self.pdb.coords = self.mol.as_array()

                nrotations = int(360/self.dtheta)

                all_torsion_atoms = sorted(self.mol.get_rotatable_torsions(include_rings=True))
                all_torions = []
                ring_torsion_atoms = []
                for t in all_torsion_atoms :
                        if t[1] in self.rings[ring_index] and t[2] in self.rings[ring_index] :
                                all_torsions.append(t)

                # Create new geometries
                geometries = [numpy.array(self.pdb.coords)]
                for bond in bonds :
                        bondlength = pdb.get_bondlength([ia+1 for ia in bond])
                        print ('Bond: ',bond, bondlength)
                
                        # Break the bonds one by one
                        broken_pdb = self.pdb.remove_bond([i+1 for i in bond])
                
                        # Find the rotatable bonds (can be done faster)
                        broken_mol = broken_pdb.get_plamsmol()
                        torsions = sorted(broken_mol.get_rotatable_torsions(include_rings=False))
                        torsions = [t[::-1] if t[0] in bond else t for t in torsions]
                        # Use only the torsions directly at the broken bond
                        torsions = [t for t in torsions if t[3] in bond]
                        for tor in torsions: print(tor)
                
                        # Create all the new geometries
                        ngeoms = len(geometries)
                        geometries = self.update_ring_geometries(geometries, broken_pdb, torsions, nrotations, bond)
                
                        # Do a check of the distance in the broken bond?
                        new_geometries = geometries[:ngeoms]
                        for ic,crd in enumerate(geometries[ngeoms:]) :
                                broken_pdb.coords = crd
                                new_bondlength = broken_pdb.get_bondlength([ia+1 for ia in bond])
                                #print ('new bondlength: ',new_bondlength)
                                if new_bondlength < (self.threshold*bondlength) :
                                        new_geometries.append(crd)
                        geometries = new_geometries

                return geometries

        def update_ring_geometries (self,geometries, pdb, torsions, nrotations, bond) :
                """
                Add new geometries by rotation
                """
                for i,tor in enumerate(torsions) :
                        new_geometries = []
                        for crd in geometries :
                                pdb.coords = copy.deepcopy(crd)
                                for irot in range(nrotations-1) :
                                        # Rotate around tor twice by 120 degrees
                                        pdb.change_torsion(tor,self.dtheta)
                                        new_crd = copy.deepcopy(pdb.coords)
                                        new_geometries.append(new_crd)
                        geometries += new_geometries
                return geometries

        def print_torsion_angles (self, geometries) :
                """
                Print the torsion angles for all these geometries

                FIXME: This should be moved to the conformer object
                """
                for i,crd in enumerate(geometries) :
                        dih = []
                        for tor in self.torsion_atoms :
                                tor_ = [t+1 for t in tor]
                                dih.append(dihedral(*crd[tor],unit='degree'))
                        print ('%8i %10.1f %10.1f %10.1f'%(i,dih[0],dih[1],dih[2]))
                
                print ('nconfs: ',len(geometries))

                return geometries

