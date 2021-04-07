#!/usr/bin/env python

"""
Author: Rosa Bulo, 2021

Class holding a z-matrix with reordered atoms
"""

import sys
import os
import copy
import numpy
from scm.plams import angle
from scm.plams import dihedral
from scm.plams import axis_rotation_matrix

__all__ = ['ZMatrix']

class ZMatrix :
        """
        Holds the zmatrix for a molecule, with reordered atoms
        """
        def __init__ (self) :
                """
                Initiates an instance of the ZMatrix class
                """
                self.mol = None
                self.atom_list = None
                self.connectivity = None
                self.indices = None

                self.backbone = None
                self.terminal_atoms = None

                self.angle_units = 'radian'

        def prepare_state (self, mol) :
                """
                Generates optimal z-martrix by reordering the atoms
                """
                self.mol = mol
                self.backbone = self.mol.find_main_chain()
                self.terminal_atoms = self.mol.get_terminal_atoms()
                self.create_connectivity()
                self.get_values()

        def get_values (self, coords=None) :
                """
                Get the values in the zmatrix

                Note: Angles are in radians
                """
                mol = self.mol.copy()
                if coords is not None :
                        mol.from_array(coords)
                else :
                        coords = mol.as_array()

                internal_coords = []
                for i,at in enumerate(self.atom_list) :
                        atoms = [mol.atoms[at]]
                        conect = self.connectivity[i]
                        atoms += [mol.atoms[at] for at in conect]
                        values = []
                        dist = 0.
                        if len(conect) > 0 :
                                dist = mol.atoms[at].distance_to(mol.atoms[conect[0]])
                        values.append(dist)
                        phi = 0.
                        if len(conect) > 1 :
                                vec1 = coords[at] - coords[conect[0]]
                                vec2 = coords[conect[1]] - coords[conect[0]]
                                phi = angle(vec1, vec2, result_unit=self.angle_units)
                        values.append(phi)
                        theta = 0.
                        if len(conect) > 2 :
                                theta = dihedral(atoms[0].coords, atoms[1].coords, atoms[2].coords, atoms[3].coords, unit=self.angle_units)
                        values.append(theta)
                        internal_coords.append(values)
                internal_coords = numpy.array(internal_coords)

                return internal_coords

        def get_cartesian_coords (self, zmat_values) :
                """
                Convert z-matrix values to Cartesian coordinates

                * ``zmat_values`` -- All angles are in radians
                """
                coords = numpy.zeros((len(self.mol),3))

                # Create the internal coordinate axes
                v_ij, v_m, v_n = numpy.identity(3)

                coord = numpy.zeros(3)
                for i,iat in enumerate(self.atom_list) :
                        conect = self.connectivity[i]
                        d,a,t = zmat_values[i]

                        if len(conect) == 0 :
                                coords[iat] = 0.
                                continue
        
                        i = conect[0]
                        # Create the internal coordinate axes
                        v_ij, v_m, v_n = numpy.identity(3)       
                        if len(conect) > 1 : 
                                j = conect[1]
                                # Get the vector from i to j
                                v_ij = coords[j] - coords[i]
                                v_ij = v_ij / numpy.sqrt((v_ij**2).sum())
        
                        if len(conect) > 2 :
                                k = conect[2]
                                # Get the vector from j to k
                                v_jk = coords[k] - coords[j]
                
                                # Get the vector orthoganl to the plane ijk
                                v_n  = numpy.cross(v_ij,v_jk)
                                v_n = v_n / numpy.sqrt((v_n**2).sum())
                
                                # Get the vector orthogonal to the plane though i, j, and v_n
                                v_m = numpy.cross(v_n,v_ij)
                                v_m = v_m / numpy.sqrt((v_m**2).sum())
        
                        # Now get the coordinates in the coordinate system of v_ij, v_m, and v_n
                        coord[0] = d * numpy.cos(a)
                        coord[1] = d * numpy.sin(a) * numpy.cos(t)
                        coord[2] = d * numpy.sin(a) * numpy.sin(-t)
        
                        # translate it to the actual xyz coordinate system
                        coords[iat] = coords[i] + (coord[0]*v_ij) + (coord[1]*v_m) + (coord[2]*v_n)

                return coords

        def print_zmatrix (self, zmat_values) :
                """
                Write the z-matrix
                """
                elements = [at.symbol for at in self.mol.atoms]
                block = ''
                conns = numpy.zeros(3)
                for i,el in enumerate(elements) :
                        block += '%8s '%(el)
                        conns[:len(self.connectivity[i])] = self.connectivity[i]
                        for iat in conns :
                                block += '%5i '%(iat)
                        for v in zmat_values[i] :
                                block += '%20.10f '%(v)
                        block += '\n'
                return block

        def create_connectivity (self) :
                """
                Create the new atom ordering and connectivity
                """
                # Get the starting atom (will fail for methane)
                start_index = self.get_first_atom()
                if start_index is None :
                        print ('Warning: No zmatrix could be generated')
                        return
                start = self.backbone[start_index]

                # Get the first four atoms in the new z-matrix
                atom_list = self.backbone[start_index:start_index+4]
                connectivity = [[], atom_list[:1][::-1], atom_list[:2][::-1], atom_list[:3][::-1]] # Use internal numbering instead?
                level_dictionary = {0:[atom_list[0]],1:[atom_list[1]],2:[atom_list[2]],3:[atom_list[3]]}

                # Now loop over neighbors starting with the start atom
                level = 0
                while level in level_dictionary :
                        for at in level_dictionary[level] :
                                iat = atom_list.index(at)
                                neighbors = [self.mol.index(n)-1 for n in self.mol.neighbors(self.mol.atoms[at])]
                                for at_next in neighbors :
                                        if at_next in atom_list : continue
                                        # This atom we will append to the z-matrix
                                        atom_list.append(at_next)
                                        if not level+1 in level_dictionary :
                                                level_dictionary[level+1] = []
                                        level_dictionary[level+1].append(at_next)
                                        conect = [at] + connectivity[iat][:2]
                                        # Now what if there is not enough connectivity here (only happens for first two atoms)?
                                        if len(conect) < 3 :
                                                conect = atom_list[iat:iat+3]
                                        connectivity.append(conect)
                        level += 1

                self.atom_list = atom_list
                self.connectivity = connectivity
                self.indices = [self.atom_list.index(i) for i in range(len(self.mol))]

        def get_first_atom (self) :
                """
                Get the starting atom (heavy atom with highest valence?)

                Note: May change self.backbone!
                """
                # Get the starting atom (heavy atom with highest valence?)
                valences = []
                for at in self.backbone : 
                        nbs = [n for n in self.mol.neighbors(self.mol.atoms[at]) if not self.mol.index(n)-1 in self.terminal_atoms]
                        valences.append(len(nbs))
                indices = numpy.array(valences).argsort()

                if len(self.backbone) < 4 :
                        # I have to create a new backbone in this case
                        self.extend_backbone()
                        if len(self.backbone) < 4 :
                                # Only methane
                                return None
                for ind in indices :
                        # The chain has to have 3 more atoms following the starting atom
                        if len(self.backbone) > ind+3 :
                                start_index = ind
                                break
                        elif ind >= 3 :
                                # Invert the backbone
                                self.backbone = self.backbone[::-1]
                                start_index = len(self.backbone)-ind-1
                                break
                return start_index

        def extend_backbone (self) :
                """
                Extend the backbone at both ends
                """
                pos = 0
                neighbor = self.find_neighbor_to_backbone(pos)
                if neighbor is not None :
                        self.backbone = [neighbor] + self.backbone
                pos = -1
                neighbor = self.find_neighbor_to_backbone(pos)
                if neighbor is not None :
                        self.backbone = self.backbone + [neighbor]

        def find_neighbor_to_backbone (self, pos) :
                """
                Find extension atom to backbone

                * ``pos`` -- Integer representing the head (0) or the tail (-1) of the backbone
                """
                neighbor = None
                neighbors = [self.mol.index(n)-1 for n in self.mol.neighbors(self.mol.atoms[self.backbone[pos]])]
                neighbors = [atn for atn in neighbors if not atn in self.backbone]
                if len(neighbors) > 0 :
                        neighbor = neighbors[0]
                return neighbor

