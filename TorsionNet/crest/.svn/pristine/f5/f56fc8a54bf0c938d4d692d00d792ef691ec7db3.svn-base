#!/usr/bin/env python

"""
Authors: 
        - Ravi Ramautar, 2020
        - David Wessels, 2021

Class for the generation of conformers from CREST metadynamics simulations
"""

import sys
import numpy
from scm.plams import Molecule
from scm.plams import AMSJob
from scm.plams import Settings
from scm.plams import add_to_class
from scm.flexmd import MDMolecule
from scm.flexmd import pdb_from_plamsmol
from scm.flexmd import locate_rings

#__all__ = ['compute_flexibility_factor','predict_nconformers','get_rotatable_bonds','get_nrotatable_bonds']

@add_to_class(Molecule)
def compute_flexibility_factor (self, engine_settings=None, nproc=1) :
        """
        Compute the data needed for the flexibilty factor, and then combine into mu

        * ``engine_settings`` -- PLAMS Settings object:
                 engine_settings = Settings()
                 engine_settings.DFTB.Model = 'GFN1-xTB'
        """
        self.guess_bonds()

        # Get the Mayer bond order
        bnd_pairs, B_ab = self._get_Mayer_bond_orders(engine_settings, nproc)

        # Get number of neighbors for the atoms in the bonds
        nbs_list = [len(self.neighbors(atom)) for atom in self]
        nneighbours = numpy.array([(nbs_list[at1],nbs_list[at2]) for (at1,at2) in bnd_pairs])

        # Get the ringfactors
        R = self._get_ringfactors(bnd_pairs)

        # Calculating flexibility        
        mu = flexibility_factor_from_bonddata (B_ab, nneighbours, R)
        return mu

@add_to_class(Molecule)
def predict_nconformers (self, angle_estimation=True, no_fused_ring_bonds=True) :
        """
        Use the rotatable bonds to predict the number of conformers
        """
        nrot_bonds2 = 0
        nrot_bonds_all = self.get_nrotatable_bonds(no_fused_ring_bonds)
        nrot_bonds = nrot_bonds_all
        if angle_estimation :
            nrot_bonds2 = self._get_nrotatable_bonds_sp2()
            nrot_bonds = nrot_bonds_all - nrot_bonds2
        return int(3**nrot_bonds * 2**nrot_bonds2)

@add_to_class(Molecule)
def get_nrotatable_bonds (self, no_fused_ring_bonds=True, include_rings=True) :
    """
    Find the number of rotatable bonds
    """
    bond_pairs = self.get_rotatable_bonds(no_fused_ring_bonds,include_rings)

    ring_factors = self._get_ringfactors(bond_pairs)
    nrotbonds = ring_factors.sum()

    return nrotbonds

@add_to_class(Molecule)
def get_rotatable_bonds (self, no_fused_ring_bonds=True, include_rings=True):
    """
    Find all rotatable bonds
    """
    self.guess_bonds()
    self.label(keep_labels=True)
    rings = self.get_rings()

    # All rotatable bonds
    bond_pairs = []

    for bond in self.bonds:
        indices = [ind-1 for ind in self.index(bond)]

        if in_fused_ring(rings, indices[0], indices[1]) and no_fused_ring_bonds :
            continue

        if not include_rings :
                ring_bond = False
                for ring in rings :
                        if indices[0] in ring and indices[1] in ring :
                                ring_bond = True
                if ring_bond :
                        continue

        atoms = [self.atoms[i] for i in indices]
        terminal = False
        symmetric = False
        for i, atom in enumerate(bond):
            atom2 = [at for at in atoms if not at is atom][0]
            neighbors = [neighbor for neighbor in self.neighbors(atom) if neighbor is not atom2]

            if len(neighbors) == 0:
                terminal = True
            elif len(neighbors) == 3:
                # If all three subgroups are the same rotation will have no effect
                if len(set([n.IDname for n in neighbors]))==1 :
                    symmetric = True

        if not terminal and not symmetric and bond.order==1:
            bond_pairs.append(indices)

    return bond_pairs

@add_to_class(Molecule)
def get_rotatable_torsions (self, no_fused_ring_bonds=True,include_rings=True):
    """
    Define the four atoms in the torsion for each rotatable bond
    """
    labels = None
    if hasattr(self.atoms[0],'IDname') :
        labels = numpy.array([at.IDname for at in self.atoms])

    bonds = self.get_rotatable_bonds(no_fused_ring_bonds,include_rings)
    # Define the torsion
    masses = numpy.array([at.mass for at in self.atoms])
    torsions = []
    for bond in bonds :
        torsion = self.unique_torsion_from_bond(bond, masses, labels)
        torsions.append(torsion) 

    return torsions

@add_to_class(Molecule)
def unique_torsion_from_bond (self, bond, masses=None, labels=None) :
        """
        Return a unique set of for torsion atoms for this bond

        * ``bond``   -- Atom indices for this bond
        * ``masses`` -- A list with the mass of each atom in self.atoms
        * ``labels`` -- A numpy array with the unique label of each atom in self.atoms
        """
        if masses is None :
            masses = numpy.array([at.mass for at in self.atoms])

        one_four = []
        for iat in bond :
            neighbors = [self.index(n)-1 for n in self.atoms[iat].neighbors() if not self.index(n)-1 in bond]
            # Now find the heaviest neighbor
            max_mass = masses[neighbors].max()
            indices = numpy.where(masses[neighbors]==max_mass)[0]
            heavy_neighbors = [neighbors[i] for i in indices]
            if labels is not None :
                neighbor = heavy_neighbors[labels[heavy_neighbors].argmin()]
            else :
                neighbor = heavy_neighbors[0]
            one_four.append(neighbor)
        torsion = [one_four[0]] + bond + [one_four[1]]
        return torsion

@add_to_class(Molecule)
def get_rings (self) :
    """
    Return the rings
    """
    pdb = pdb_from_plamsmol(self)
    mdmol = MDMolecule(pdb=pdb)
    rings = numpy.array(locate_rings(mdmol,[i for i in range(len(self))]))
    return rings

@add_to_class(Molecule)
def _get_Mayer_bond_orders (self, engine_settings=None, nproc=1) :
    """
    Get the Mayer bond orders by performing a DFTB single point
    """
    # Run the single point with GFN1xTB
    result = self._get_dftb_result(engine_settings, nproc)
    enginename = result.engine_names()[0]

    # Locate the property on the KF file
    keys = result.get_rkf_skeleton(enginename)['Properties']
    properties = {}
    for key in keys :
        if 'Subtype' in key :
            propname = (result.readrkf('Properties',key,enginename)).strip()
            num = int(key.split('(')[1].split(')')[0])
            properties[propname] = num
    # Read the property
    bonded_atoms = numpy.array(result.readrkf('Properties','Value(%i)'%(properties['Bonded Atoms (Mayer)']),enginename)) - 1
    bonded_atoms = bonded_atoms.reshape((int(len(bonded_atoms)/2),2))
    bond_orders = numpy.array(result.readrkf('Properties','Value(%i)'%(properties['Bond Orders (Mayer)']),enginename))
    return bonded_atoms, bond_orders

@add_to_class(Molecule)
def _get_dftb_result (self, engine_settings=None, nproc=1) :
        """
        Run a GFN1xTB job to get the bond orders and other data
        """
        if engine_settings is None :
                        engine_settings = Settings()
                        engine_settings.DFTB.Model = 'GFN1-xTB'

        settings = Settings()
        settings.runscript.nproc = nproc
        settings.input.ams.Task = 'SinglePoint'
        engine_name = list(engine_settings.keys())[0]
        settings.input[engine_name] = engine_settings[engine_name]
        settings.input.ams.Properties.BondOrders = True
        job = AMSJob(molecule=self, settings=settings)
        result = job.run()
        return result

@add_to_class(Molecule)
def _get_ringfactors (self, bnd_pairs) :
        """
        Compute the ringfactor for each bond
        """
        rings = self.get_rings()
        
        ring_factors = []
        for bond in bnd_pairs :
                ringsize = 0
                for ring in rings :
                        if bond[0] in ring and bond[1] in ring :
                                ringsize = len(ring)
                                break
                ring_factors.append(ringfactor_from_ringsize(ringsize))
        ring_factors = numpy.array(ring_factors)
        return ring_factors
        
@add_to_class(Molecule)
def _get_nrotatable_bonds_sp2 (self):
    """
    Find all rotatable bonds that have an approximately flat geometry
    """
    self.guess_bonds()
    self.label(keep_labels=True)

    # Rotatable bonds with base 2 
    bond_pairs = []

    for bond in self.bonds:
        indices = [ind-1 for ind in self.index(bond)]

        atoms = [self.atoms[i] for i in indices]
        terminal = False
        aromatic = False
        for i, atom in enumerate(bond):
            atom2 = [at for at in atoms if not at is atom][0]

            if len(self.neighbors(atom)) == 1:
                terminal = True
            elif self._heteroatom_connected_aromatic_ring(atom,bond,atom2) :
                aromatic = True

        if not terminal and bond.order==1 and aromatic:
            bond_pairs.append(indices)

    return len(bond_pairs)

@add_to_class(Molecule)
def _heteroatom_connected_aromatic_ring(self, atom, cur_bond, atom2):
    """
    Checks if atom is directly bonded to an aromatic ring
    """
    aromatic_bools = [bond.is_aromatic() for bond in atom.bonds if bond is not cur_bond]

    if any(aromatic_bools) == True and atom2.symbol in ["N", "O"] and not self.in_ring(atom2) :
        return True
    return False

@add_to_class(Molecule)
def find_main_chain (self) :
        """
        All atoms in the system will be either main chain M, side chain S, or end point E

        Note: A single molecule needs to be passed (all atoms conected)
        """
        nats = len(self)

        # Create a connection table
        conect = self.get_connection_table()

        # Mark the end points and mark the cross sections
        cross = []
        ends = []
        for at in range(len(self)) :
                neighbors = conect[at+1]
                if len(neighbors) == 1 :
                        ends.append(at)
                if len(neighbors) > 2 :
                        cross.append(at)

        if len(cross) == 0 :
                path = self.get_shortest_path_dijkstra(ends[0],ends[1])[0]
                return path

        tot_paths = []
        for c_at in cross :
                #print('source c_at',c_at)
                neighbors = numpy.array(conect[c_at+1]) - 1
                #print('neighbors: ',neighbors)
                paths = []
                lengths = []
                for n in neighbors :
                        paths.append([])
                        lengths.append([])
                for e_at in ends :
                        #print ('target e_at: ',e_at)
                        path = self.get_shortest_path_dijkstra(c_at,e_at)[0]
                        if len(path) == 0 : continue
                        #print('path: ',path)
                        # Path goes from target to source
                        # Now assign the path to a neighbor
                        for i,n in enumerate(neighbors) :
                                if n == path[-2] :
                                        paths[i].append(path)
                                        lengths[i].append(len(path))
                #print ('paths', paths)
                # Now find the two neighbors that have the longest paths (also the paths themselves)
                max_paths = []
                max_path_indices = []
                for len_list in lengths :
                        max_path = max(len_list)
                        max_paths.append(max_path)
                        max_path_index = len_list.index(max_path)
                        max_path_indices.append(max_path_index)
                # Sort the path indices based on the lenth of the paths
                zipped = sorted(list(zip(max_paths,max_path_indices,range(len(paths)))))
                zipped = zipped[-2:]
                path_indices = [y for x,y,z in zipped]
                neighbor_indices = [z for x,y,z in zipped]
                # Now connect the two paths into a single longest path
                totpath = []
                for n_ind, p_ind in zip(neighbor_indices,path_indices) :
                        half = paths[n_ind][p_ind]
                        if len(totpath) > 0 :
                                half.reverse()
                                half = half[1:]
                        totpath += half
                #print('totpath: ',totpath)
                tot_paths.append(totpath)
        lengths = []
        for path in tot_paths :
                lengths.append(len(path))
        longest = max(lengths)
        index_longest_path = lengths.index(longest)

        return tot_paths[index_longest_path][1:-1]

@add_to_class(Molecule)
def get_shortest_path_dijkstra (self, source, target) :
        """
        Returns shortest path from target to source
        """
        nats = len(self)
        atlist = range(nats)

        # Create a connection table
        conect = self.get_connection_table()

        huge = 100000.
        dist = {}
        previous = {}
        for v in atlist :
                dist[v] = huge
                previous[v] = []
        dist[source] = 0

        Q = [source]
        for at in atlist :
                if at != source :
                        Q.append(at)

        while len(Q) > 0 :
                # vertex in Q with smallest distance in dist
                u = Q[0]
                if dist[u] == huge :
                        print('Path to target not found')
                        return []
                u = Q.pop(0)
                if u == target :
                        #print('target found')
                        break

                # Select the neighbors of u, and loop over them
                neighbors = numpy.array(conect[u+1])-1
                for v in neighbors :
                        if not v in Q :
                                continue
                        alt = dist[u] + 1.
                        if alt == dist[v] :
                                previous[v].append(u)
                        if alt < dist[v] :
                                previous[v] = [u]
                                dist[v] = alt
                                # Reorder Q
                                for i,vertex in enumerate(Q) :
                                        if vertex == v :
                                                ind = i
                                                break
                                Q.pop(ind)
                                for i,vertex in enumerate(Q) :
                                        if dist[v] < dist[vertex] :
                                                ind = i
                                                break
                                Q.insert(ind,v)

        bridgelist = [[u]]
        d = dist[u]
        for i in range(int(d)) :
                paths = []
                for j,path in enumerate(bridgelist) :
                        prevats = previous[path[-1]]
                        for at in prevats :
                                newpath = path+[at]
                                paths.append(newpath)
                bridgelist = paths

        return bridgelist

@add_to_class(Molecule)
def get_connection_table (self) :
        """
        Gets a connection table for the atoms
        """
        conect = {}
        for iat,atom in enumerate(self.atoms) :
                neighbors = self.neighbors(atom)
                if len(neighbors) > 0 :
                        conect[iat+1] = [self.index(neighbor) for neighbor in neighbors]
        return conect

@add_to_class(Molecule)
def get_terminal_atoms (self) :
        """
        Returns a list of terminal atoms (only one bond)
        """
        return [iat for iat,at in enumerate(self.atoms) if len(self.neighbors(at))==0]

@add_to_class(Molecule)
def get_atoms_smallest_ring (self) :
        """
        Get the atoms in the smallest ring to be used to overlay all conformers
        """
        # Get the atoms in small rings, as they can be expected to be most rigid
        rings = self.get_rings()
        rings = [ring for ring in rings if len(ring) <= 6]
        atoms = []
        if len(rings) > 0 :
                ind = [len(ring) for ring in rings].index(min([len(ring) for ring in rings]))
                atoms = rings[ind]
        # If there are no rings, use all carbon atoms
        if len(atoms) == 0 : atoms = [i for i in range(len(self)) if self.atoms[i].symbol=='C']
        return atoms

def ringfactor_from_ringsize (ringsize) :
        """
        Flexibility factor from ringsize
        """
        if ringsize > 10 :
                ringsize = 10
        ringfactors = [1.,0,0,0,0.1,0.1,0.2,0.3,0.5,0.8,0.9]
        return ringfactors[ringsize]

def flexibility_factor_from_bonddata (B_ab, nneighbors, R) :
        """
        Compute the flexibility factor
        """
        if len(R) == 0 :
                # If there are no bonds in the system, lets say flexibility equals 1
                return 1.
        mu = ( (1-numpy.exp(-5*(B_ab-2)**10))**2 * (4/(nneighbors[:,0]*nneighbors[:,1])) * (R)**2 ).sum()
        mu = numpy.sqrt(1/len(B_ab)) * numpy.sqrt(mu)
        return mu

def in_fused_ring(rings, i, j):
    """
    Checks if two atoms (i,j) form a bond that is shared by multiple rings
    """
    return sum(i in ring_lis for ring_lis in rings)+sum(j in ring_lis for ring_lis in rings) == 4

