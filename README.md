# SiMDel

**Si**mple **m**olecular **d**ynamics **e**xpandable **l**ibrary is a simple library for molecular dynamics.

### Introduction

Allows you to prepare and calculate molecular dynamics directly from python. It does not require editing files or converting to other formats,
completely seamless molecular dynamics!

**Main differences from other libraries**:

- Everything is typed, and all stages are always functions with a clear signature.
- Simple and fast aggregation of information about the molecular system in the form of `pandas.DataFrame`
- Modular library structure - easy to add new functions, MD engines or complex pipelines
- A universal templates for the MD computing - pipelines

### Installation

The library requires several types of dependencies: pip, conda, GROMACS, PLUMED. You may not install everything, then part of the library will be inactive.

The installation order does not need to be changed

#### 1. GROMACS

_Mandatory_

1. Plain without PLUMED ([see documentation](https://manual.gromacs.org/documentation/current/install-guide/index.html))
2. Regular c PLUMED ([PLUMED flag](https://manual.gromacs.org/documentation/current/install-guide/index.html#building-with-plumed-support))
3. Via CONDA - there is no GPU support, it is impossible to connect PLUMED:

   ```
   conda install gmx
   ```

#### 2. PLUMED

_Required for: metadynamics_

Modern GROMACS has a PLUMED-enabled build flag. Side of the source code - [see documentation](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html)

#### 3. Conda dependencies

_Required for: openff parameterization, building a transition graph for FEP_

```
conda install -f conda.yml -y
```

#### 4. Pip dependencies

_Mandatory_

1. UV. The installation takes place in .venv, because uv does not know how to install packages in an arbitrary location.

   ```
   uv sync --inexact
   ```

2. Poetry

   ```
   poetry install
   ```

### 5. PMX package

_Required for: FEP_

It is outdated until the update is done, the only way is to install the pmx package from this project (will be updated soon)

```
cd pmx
pip install .
```

## Working principle

It is based on the structure of the GROMACS molecular system. This is a relational database, and thanks to pandas, you can quickly perform complex manipulations with systems and calculate any parameters. The system is immutable. Together with other main classes, it ensures the consistency of the library.

All manipulations with the system are functions that return the new system as a result of work. The only side effect is the creation of a folder for intermediate files.

Individual functions are combined in the pipeline stages, and these stages also have only one side effect - the creation of working folders.

## P.S.

_The simplest mechanics of video games are shooting and moving on wasd._

Many libraries are tailored to perform a specific task: parameterization, MD simulation, analysis, and assembly of a molecular system from individual molecules. Thus, you have to assemble your own conveyor for yourself. For a researcher, this is a technical task that must meet the minimum requirements. This leads to a loss of versatility, scalability, and problems adapting different tools.

The main problem is inconsistency, different interface, and how libraries work. We have to build converters every time and keep in mind the features of the tool. When converting formats, data can be lost, building wrappers takes time, and using ready-made frameworks does not allow you to add your own functions without a deep understanding of the framework design.

Therefore, the idea arose to create the simplest library possible, which will connect individual tools with a single interface. And there is nothing simpler than calling a function for arguments! This will allow you to simply add new functions and argument types to the common pot. That's how `SiMDel` appeared
