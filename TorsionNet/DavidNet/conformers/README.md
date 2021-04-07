SCM Conformers
==============

Python library for conformer generation
---------------------------------------

The conformers package holds a collection of tools for the generation of conformer sets for a molecule.
The main tools for conformer generation in the package are:

* CREST conformer generation (DOI: 10.1039/c9cp06869d)
* RDKit ETKDG conformer generation (DOI: 10.1021/acs.jcim.5b00654)

The top level classes in the conformers package are the UniqueConformers classes:
* UniqueConformersCrest
* UniqueConformersAMS
* UniqueConformersTFD

The main task of the conformer class is to hold the conformers of a molecule and prune/filter out duplicates.
There are several ways duplicates can be defined :
* CREST duplicate recognition (DOI: 10.1039/c9cp06869d, UniqueConformersCrest)
* based on interatomic distances and torsion angles (UniqueConformersAMS)
* based on the Torsion Fingerprint Difference (DOI: 10.1021/ci2002318, UniqueConformersTFD)

In addition to duplicate pruning, the UniqueConformers classes are also linked to a default conformer generation method 
(which can be changed by the user) and to a geometry optimization method (also changeable by the user).

An empty instance of the conformers object is created,
and then conformers can be added to it, read from file, or generated.
The conformers in the set can then be manipulated and compared.

Simple example
--------------

    from scm.plams import Molecule
    from scm.plams import init, finish
    from scm.conformers import UniqueConformersAMS

    # Set up the molecular data
    mol = Molecule('mol.xyz')
    conformers = UniqueConformersAMS()
    conformers.prepare_state(mol)
   
    # Set up PLAMS file handling
    init()

    # Generate the conformers using the RDKit ETKDG method
    # This method includes geometry optimization and pruning
    conformers.generate(nprocs_per_job=1, nprocs=4)

    finish()

    # Write the results to file
    print(conformers)
    conformers.write()
