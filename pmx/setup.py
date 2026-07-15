"""pmx: a toolkit for free-energy calculation setup/analysis and
biomolecular structure handling.

For installation type the command::
    python setup.py install
or
    pip install .
"""

import json
from distutils.core import Command
from pathlib import Path
from string import Template

from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def readme():
    with open("README.rst") as f:
        return f.read()


# ----------
# Extensions
# ----------
pmx = Extension(
    "pmx._pmx",
    libraries=["m"],
    include_dirs=["src/pmx/extensions/pmx"],
    sources=[
        "src/pmx/extensions/pmx/Geometry.c",
        "src/pmx/extensions/pmx/wrap_Geometry.c",
        "src/pmx/extensions/pmx/init.c",
        "src/pmx/extensions/pmx/Energy.c",
    ],
)

xdrio = Extension(
    "pmx._xdrio",
    libraries=["m"],
    include_dirs=["src/pmx/extensions/xdr"],
    sources=[
        "src/pmx/extensions/xdr/xdrfile.c",
        "src/pmx/extensions/xdr/xdrfile_trr.c",
        "src/pmx/extensions/xdr/xdrfile_xtc.c",
    ],
)
extensions = [pmx, xdrio]

# -----
# CMD
# -----
SHORT_VERSION_PY = Template(
    """\
import json

version_json = '''
$version_json
'''  # END VERSION_JSON


def get_versions():
    return json.loads(version_json)
"""
)

VERSIONS = {
    "version": "1.0.0",
    "full-revisionid": "",
    "dirty": True,
    "error": None,
    "date": "2024-02-01T14:27:54+0100",
}


class CmdVersion(Command):
    description = "report generated version string"
    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


class VersioneerConfig:
    """ """

    def __init__(self):
        self.VCS = "git"
        self.style = "pep440"
        self.versionfile_source = "src/pmx/_version.py"
        self.versionfile_build = "pmx/_version.py"
        self.tag_prefix = ""
        self.parentdir_prefix = "pmx-"


class CmdBuildPy(build_py):
    def run(self):
        build_py.run(self)

        # now locate _version.py in the new build/ directory and replace
        # it with an updated value
        cfg = VersioneerConfig()
        if cfg.versionfile_build:
            target_versionfile = Path(self.build_lib) / cfg.versionfile_build
            target_versionfile.write_text(
                SHORT_VERSION_PY.substitute(version_json=json.dumps(VERSIONS))
            )


class CmdSdist(sdist):
    def run(self):
        self.distribution.metadata.version = VERSIONS["version"]
        return sdist.run(self)

    def make_release_tree(self, base_dir, files):
        cfg = VersioneerConfig()
        sdist.make_release_tree(self, base_dir, files)

        # now locate _version.py in the new base_dir directory
        # (remembering that it may be a hardlink) and replace it with an
        # updated value
        target_versionfile = Path(base_dir) / cfg.versionfile_source
        target_versionfile.write_text(
            SHORT_VERSION_PY.substitute(version_json=json.dumps(VERSIONS))
        )


CMDS = {
    "version": CmdVersion,
    "build_py": CmdBuildPy,
    "sdist": CmdSdist,
}

# -----
# Setup
# -----
setup(
    name="pmx",
    version=VERSIONS["version"],
    cmdclass=CMDS,
    description="Toolkit for free-energy calculation setup/analysis "
    "and biomolecular structure handling",
    long_description=readme(),
    classifiers=[
        "Development Status :: Beta",
        "License :: OSI Approved :: GNU General Public License (LGPL)",
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    url="https://github.com/deGrootLab/pmx/tree/develop",
    author="Daniel Seeliger, Vytautas Gapsys",
    author_email="d.seeliger@gmx.net, vytautas.gapsys@gmail.com",
    license="LGPL-3.0",
    packages=["pmx"],
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    ext_modules=extensions,
    setup_requires=[],
    tests_require=["pytest"],
    install_requires=["numpy", "scipy", "matplotlib", "future"],
    python_requires=">=3.10",
    entry_points={"console_scripts": ["pmx = pmx.scripts.cli:entry_point"]},
)
