import os
import glob
import json
import pickle
import logging
import numpy as np

import gym
from torch_geometric.data import Batch

import crest
from scm.plams import Molecule, to_rdmol
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

from .utils import mol2vecskeletonpoints, print_torsions, ConformerGeneratorCustom, \
                    prune_last_conformer, calc_gibbs_norm_values



confgen = ConformerGeneratorCustom(max_conformers=1,
                             rmsd_threshold=None,
                             force_field='mmff',
                             pool_multiplier=1)


class GibbsMolEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg):
        super(GibbsMolEnv, self).__init__()

        # Boolean properties
        self.eval = cfg.eval
        self.temp_normal = cfg.temp_normal
        self.gibbs_normalize = cfg.gibbs_normalize
        self.pruning_thresh = cfg.pruning_thresh
        self.plams_mol_bool = cfg.plams_mol 
        self.plams_tor_bool = cfg.plams_tor
        
        # Curricula properties
        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

        # Env properties
        self.max_steps = 200

        # Load files 
        self.file_ext = cfg.file_ext
        self.folder_name = cfg.folder_name
        self.all_files = self.load_files(cfg.sort_by_size)
        print("Loaded following files: ", self.all_files)

        # Determine molecule and rotatable bonds
        self.mol, self.conf = self.choose_molecule()
        self.nonring_torsions, ring_torsions = self.get_possible_torsions()

        # calc_gibbs_norm_values(self.mol, num_confs = 500)

        # print("Nonring torsions: ", self.nonring_torsions, "\n\n")
        logging.info(f'rbn: {len(self.nonring_torsions)}')

        # Properties regarding reward calculations
        self.delta_t = []
        self.current_step = 0
        self.seen = set()
        self.energys = []
        self.zero_steps = 0
        self.repeats = 0
        self.backup_energys = []



    def load_files(self, sort_by_size):
        """Load molecules from json files
            
            folder_name:  String containing the initial molecule directory.
            sort_by_size: Boolean which specifies if molecules are sorted by size.

            returns: Sorted list with molecule files.

        """
        tmp = f'{self.folder_name}*.'+ self.file_ext
        all_files = glob.glob(tmp)

        if '/' in self.folder_name:
            self.folder_name = self.folder_name.split('/')[0]

        if sort_by_size:
            all_files.sort(key=os.path.getsize)
        else:
            all_files.sort()

        return all_files


    def get_possible_torsions(self):
        """ Get all possible torsions where torsions within rings are seperately kept.
            Function has two different ways of retrieving the torsions based on the different file extensions.
            For the file extension json there is made use of rdkit, and for xyz there is made use of the SCM package.

            returns:
                nonring - List containing all torsions (ids of 4 consecutive atoms) which are not in a ring.
                ring    - List containing all torsions in a ring.
        """
        if self.plams_tor_bool:
            return self.mol.get_rotatable_torsions(), None
        else:
            nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
            nonring = [list(atoms[0]) for atoms, ang in nonring]
            return nonring, ring


    def choose_molecule(self):
        """ 
            Chooses a molecule following a curricula, and adds all relevant
            information into the environent. A molecule is only chosen when
            a conformer can be created.

        """
        if self.file_ext == "xyz":
            return self.choose_molecule_xyz()
        elif self.file_ext == "json":
            return self.choose_molecule_json()

    def choose_molecule_xyz(self):
        # Should find a way to determine inv_temp, standard energy and total energy. 
        file = self.curricula_choice()
        plams_mol = Molecule(file)
        plams_mol.guess_bonds()
        plams_mol.label(keep_labels=True)

        rd_mol = to_rdmol(plams_mol, sanitize=True, properties=True, assignChirality=False)
        rd_mol = Chem.AddHs(rd_mol)

        conf = rd_mol.GetConformer(id=0)
        res = Chem.AllChem.MMFFOptimizeMoleculeConfs(rd_mol)
        return rd_mol, conf


    def choose_molecule_json(self):
        while True:
            file = self.curricula_choice()
            print("Now loading molecule: ", file, '\n')

            with open(file) as fp:
                obj = json.load(fp)

            if 'inv_temp' in obj:
                self.temp_normal = obj['inv_temp']

            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0

            # If mol_file already in json object, 
            # otherwise load mol_file seperately from rdkit.
            if 'mol' in obj:
                mol = Chem.MolFromSmiles(obj['mol'])
                mol = Chem.AddHs(mol)
                res = AllChem.EmbedMultipleConfs(mol, numConfs=1)
                if not len(res):
                    continue
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(mol)
                conf = mol.GetConformer(id=0)
            else:
                mol = Chem.MolFromMolFile(os.path.join(self.folder_name, obj['molfile']))
                mol = Chem.AddHs(mol)
                conf = mol.GetConformer(id=0)
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(mol)

            break

        return mol, conf


    def curricula_choice(self):
        """ Chooses molecule following curricula extern """

        if self.choice_ind != 1:
            p = 0.5 * np.ones(self.choice_ind) / (self.choice_ind - 1)
            p[-1] = 0.5

            file = np.random.choice(self.all_files[0:self.choice_ind], p=p)
        else:
            file = self.all_files[0]

        return file


    def change_level(self, up_or_down):
        """ DONT KNOW WHAT IT DOESSS ...... """
        if up_or_down:
            self.choice_ind *= 2
        else:
            if self.choice_ind != 1:
                self.choice_ind = int(self.choice_ind / 2)

        self.choice_ind = min(self.choice_ind, len(self.all_files))
        

    def _get_reward(self):
        """ Calculated the reward for a action """
        if tuple(self.action) in self.seen:
            self.repeats += 1
            return 0, 0
        else:
            self.seen.add(tuple(self.action))
            current = confgen.get_conformer_energies(self.mol)[0]
            current = current * self.temp_normal
            return np.exp(-1.0 * (current - self.standard_energy)) / self.total, current

    def _get_obs(self):
        """ Creates an observation by transforming a self.mol into skeletonpoints
            and returns the observation as a rdkit Batch object. 

            returns:
        """

        data = Batch.from_data_list([mol2vecskeletonpoints(self.mol)])
        return [(data, self.nonring_torsions)]

    def step(self, action):
        # Execute one time step within the environment
        if len(action.shape) > 1:
            self.action = action[0]
        else:
            self.action = action

        self.current_step += 1

        desired_torsions = []

        for idx, tors in enumerate(self.nonring_torsions):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                logging.error('exit with debug.mol')
                exit(0)

        Chem.AllChem.MMFFOptimizeMolecule(self.mol, confId=0)
        print(self.mol)
        self.mol_appends()

        obs = self._get_obs()
        rew, energy = self._get_reward()
        self.episode_reward += rew

        energy = confgen.get_conformer_energies(self.mol)[0]
        energy = energy * self.temp_normal
        print("reward is: ", rew, "energy is: ", energy)

        # print("new state is:")
        # print_torsions(self.mol)

        info = {}
        if self.done:
            info['repeats'] = self.repeats

        info = self.info(info)

        return obs, rew, self.done, info

    @property
    def done(self):
        done = (self.current_step == self.max_steps)
        return done

    def info(self, info):
        info['choice_ind'] = self.choice_ind
        return info

    def mol_appends(self):
        pass

    def reset(self):
        self.repeats = 0
        self.current_step = 0
        self.zero_steps = 0
        self.seen = set()
        self.mol, self.conf = self.choose_molecule()

        self.episode_reward = 0
        self.nonring_torsions, ring_torsions = self.get_possible_torsions()
        logging.info(f'rbn: {len(self.nonring_torsions)}')

        obs = self._get_obs()

        # print('step time mean', np.array(self.delta_t).mean())
        # print('reset called\n\n')
        # print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)


    

class PruningSetGibbs(GibbsMolEnv):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current_energy = confgen.get_conformer_energies(self.mol)[0]
        current_energy = current_energy * 0.25 #* self.temp_normal

        print('standard', self.standard_energy)
        print('current', current_energy)

        rew = np.exp(-1.0 * (current_energy - self.standard_energy)) / self.total

        print('current step', self.current_step)
        if self.current_step > 1:
            rew -= self.done_neg_reward(current_energy)

        if self.done:
            self.backup_energys = []

        return rew, current_energy

    def done_neg_reward(self, current_energy):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        after_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)

        diff = before_total - after_total
        return diff / self.total

    def mol_appends(self):
        if self.current_step == 1:
            self.total_energy = 0
            self.backup_mol = Chem.Mol(self.mol)
            self.backup_energys = list(confgen.get_conformer_energies(self.backup_mol))
            # print('num_energys', len(self.backup_energys))
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        self.backup_energys += list(confgen.get_conformer_energies(self.mol))
        # print('num_energys', len(self.backup_energys))


class PruningSetLogGibbs(PruningSetGibbs):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        print("BACKUP",self.backup_energys, self.temp_normal)
        if self.current_step > 1:
            self.done_neg_reward()

        energys = np.array(self.backup_energys) * self.temp_normal
        print("ys", energys)

        now = np.log(np.sum(np.exp(-1.0 * (np.array(energys) - self.standard_energy)) / self.total))
        if not np.isfinite(now):
            logging.error('neg inf reward')
            now = np.finfo(np.float32).eps
        rew = now - self.episode_reward

        if self.done:
            self.backup_energys = []

        return rew, 1

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (np.array(self.backup_energys) - self.standard_energy)).sum()
        print("before", before)
        self.backup_mol, energy_args = prune_last_conformer(self.backup_mol, self.pruning_thresh, self.backup_energys)
        print("energy_args", energy_args)
        self.backup_energys = list(np.array(self.backup_energys)[np.array(energy_args)])

        assert self.backup_mol.GetNumConformers() == len(self.backup_energys)


class UniqueSetGibbs(GibbsMolEnv):
    def _get_reward(self):
        self.seen.add(tuple(self.action))
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal
        print('standard', self.standard_energy)
        print('current', current)

        rew = np.exp(-1.0 * (current - self.standard_energy)) / self.total

        # done = (self.current_step == 200)
        if self.done:
            rew -= self.done_neg_reward()
        return rew

    def done_neg_reward(self):
        before_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        before_conformers = self.backup_mol.GetNumConformers()
        self.backup_mol = prune_conformers(self.backup_mol, self.pruning_thresh)
        after_total = np.exp(-1.0 * (confgen.get_conformer_energies(self.backup_mol) - self.standard_energy)).sum()
        after_conformers = self.backup_mol.GetNumConformers()
        diff = before_total - after_total
        print('diff is ', diff)
        print(f'pruned {after_conformers - before_conformers} conformers')
        print(f'pruning thresh is {self.pruning_thresh}')
        return diff / self.total

    def mol_appends(self):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)
        if self.done and self.eval:
            
            i = 0
            while True:
                if os.path.exists(f'test_mol{i}.pickle'):
                    i += 1
                    continue
                else:
                    with open(f'test_mol{i}.pickle', 'wb') as fp:
                        pickle.dump(self.backup_mol, fp)
                    break


    

    















