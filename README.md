# SiMDEL

[Russian version](documentation/README.ru.md)

**Si**mple **M**olecular **D**ynamics **E**xpandable **L**ibrary - a library for molecular dynamics.

Setting up and running molecular dynamics simulations can be done entirely from Python. No file editing, no conversion to other formats - completely seamless molecular dynamics!

### Basic Principles

**Architecture:** The library represents a GROMACS molecular system as a set of related tabular structures (similar to a relational database), allowing `pandas` to perform complex manipulations quickly.

**Immutability:** The system object is immutable. This, together with other core classes, guarantees consistency across the library.

**Pure Functions:** All operations on the system are implemented as pure functions that return a new system. The only permitted side effect is the creation of folders for intermediate files. This design makes execution equally straightforward both locally and remotely.

**Pipelines:** Individual functions are combined into pipeline stages. Each stage also has at most one side effect - the creation of a working directory.

---

### Installation

The library requires several types of dependencies:

- **Core**: _PyPi dependencies, GROMACS_
- **Conda packages** - optional, Linux/WSL/Mac only. (_OpenFF parameterization, FEP transition graphs._ )
- **PLUMED** - optional. (_Metadynamics_)
- **PMX** - optional. (_FEP_)

> [!IMPORTANT]
>
> 1. Combinations of pip and conda are allowed, but package versions must be kept consistent to avoid conflicts.
> 2. Keep the versions of packages installed via pip and conda identical. This prevents overwriting with mismatched duplicates.
> 3. Do not change the installation order

#### 1. GROMACS

_**Required:** MD engine_

Options:

1. **Regular without** PLUMED ([see documentation](https://manual.gromacs.org/documentation/current/install-guide/index.html))
2. **Regular with** PLUMED ([PLUMED flag](https://manual.gromacs.org/documentation/current/install-guide/index.html#building-with-plumed-support))

   > [!NOTE] Built command example
   >
   > ```bash
   >  cmake .. \
   > -DGMX_BUILD_OWN_FFTW=ON \
   > -DREGRESSIONTEST_DOWNLOAD=ON \
   > -DGMX_USE_PLUMED=ON # most important - PLUMED usage
   > ```

3. **Via conda** - no GPU support, cannot enable PLUMED: **see below**

#### 2. PLUMED

_**Optional:** funnel and default metadynamics_

GROMACS must be built with PLUMED enabled. Build from source - [PLUMED installation docs](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html):

```bash
cd plumed_dir
./configure --enable-modules=all
make -j 4
make doc # this is optional and requires proper doxygen version installed
make install
```

#### 3. Conda

_**Optional:** OpenFF parameterization, transition graph construction for FEP_

Only works on Linux/WSL/Mac.

> ```bash
> conda create -n simdel_env python=3.10
> conda activate simdel_env
> conda install -f conda.yml -y
> ```

> [!NOTE] If you want GROMACS from conda, install it after creating the environment
>
> ```bash
> conda install gromacs
> ```

#### 4. Pip

_**Required:** basic classes and functions_

Options:

1. **UV**

```bash
uv pip install .
```

2. **Poetry**

```bash
poetry install
```

#### 5. PMX

_**Optional:** FEP_

> [!WARNING]
> The original package is outdated. The only working solution right now is to install the pmx fork from this project

```bash
cd pmx
pip install .
```

## P.S.

_The simplest video game mechanics are shooting and moving with WASD._

Many libraries are tailored to a specific task: parameterization, MD simulation, analysis, or assembling a molecular system from individual molecules. As a result, one has to build a custom pipeline, sacrificing versatility and scalability, and facing difficulties when integrating different tools.

The main problem is inconsistency - different interfaces, different operating principles. You have to keep track of each tool’s quirks and format conversion pitfalls; building wrappers becomes time-consuming. Ready-made frameworks rarely allow adding custom functions without deep integration into their code.

That’s why the idea emerged: the simplest possible library connecting individual tools under a single interface. What could be simpler than calling a function with arguments? That would allow anyone to easily add new functions and argument types to the common pool. This is how `SiMDel` was born.
