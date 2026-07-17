"""Molecular dynamic parameters module. See documentation:
https://manual.gromacs.org/current/user-guide/mdp-options.htm.
"""

from . import (
    constraints,
    core_mdp,
    coupling,
    external_force,
    interactions,
    others,
    qm,
    run,
    transtitions,
)
from .constraints import (
    ConstraintAlgorithm,
    Constraints,
    PullCoordGeometry,
    PullCoordType,
    RotFitMethod,
    RotType,
    WallType,
)
from .coupling import (
    Annealing,
    EnsembleTemperatureSetting,
    PCoupl,
    PCouplType,
    RefCoordScaling,
    TCoupl,
)
from .external_force import (
    DensityGuidedSimulationAtomSpreadingWeight,
    DensityGuidedSimulationSimilarityMeasure,
    Swapcoords,
)
from .interactions import (
    PBC,
    CoulombModifier,
    CoulombType,
    CutoffScheme,
    DispCorr,
    EwaldGeometry,
    LJPMEcombRule,
    VDWModifier,
    VDWType,
)
from .others import Disre, DisreWeighting, NMRDihedralConstraintMethod
from .plumed_mdp import PlumedMDP
from .qm import QMMMCp2kQmmethod
from .run import CommMode, Integrator, MTSForce
from .transtitions import (
    AWHDimCoordProvider,
    AWHGrowth,
    AWHPotential,
    AWHShareGroup,
    AWHTarget,
    CoupleLambda,
    DHDLPrintEnergy,
    FreeEnergy,
    LMCMCMove,
    LMCStats,
    LMXWeightsEquil,
    ScFunction,
    SimulatedTemperingScaling,
)


class GromacsSimulator(
    core_mdp.BaseMDP,
    others.GroupColVars,
    qm.GroupQMMMCp2K,
    external_force.GroupDensityGuidedSimulations,
    external_force.GroupComputationalElectrophysiology,
    qm.GroupQMMM,
    external_force.GroupElectricFields,
    external_force.GroupNonEquilibriumMD,
    transtitions.GroupSimulatedTempering,
    transtitions.GroupExpandedEnsembleCalculations,
    transtitions.GroupDHDL,
    transtitions.GroupFreeEnergyCouple,
    transtitions.GroupSoftFunction,
    transtitions.GroupFreeEnergy,
    others.GroupNMRRefinement,
    constraints.GroupEnforcedRotation,
    transtitions.GroupAWH,
    constraints.GroupCOMPulling,
    constraints.GroupWalls,
    constraints.GroupBonds,
    coupling.GroupVelocityGeneration,
    coupling.GroupAnnealing,
    coupling.GroupPressureCoupling,
    coupling.GroupTemperatureCoupling,
    interactions.GroupPotentialMesh,
    interactions.GroupTables,
    interactions.GroupVanDerWaals,
    interactions.GroupElectrostatics,
    interactions.GroupNeighborSearching,
    run.GroupOutControl,
    run.GroupParticleInsertion,
    run.GroupShell,
    run.GroupEnergyMinimization,
    run.GroupLangevinDynamics,
    run.GroupMotionRemove,
    run.GroupMultiTimeStepping,
    run.GroupRunControl,
):
    """GROMACS simulate parameters container."""
