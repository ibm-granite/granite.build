from typing import Union
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from pydantic import BaseModel

from gbserver.storage.target_run_storage import IStoredTargetRunStorage


class TargetSpec(BaseModel):
    target: StoredTargetRun
    step : StoredStepRun    # For now only a single step, since if we have multiple, we need to assign artifacts across N steps 
    input_artifacts: list[ArtifactRegistration]
    output_artifacts: list[ArtifactRegistration]

    def show_connections(self):
        print(f"Target {self.target.uuid} connected to build {self.target.build_id}") 
        print(f"Step   {self.step.uuid} connected to build {self.step.build_id}, target {self.step.target_id}") 
        for art in self.input_artifacts:
            print(f" Input artifact {art.uuid} connected to build {art.created_by_build_id}, target {art.created_by_target_id}, step {art.created_by_step_id}")
        for art in self.output_artifacts:
            print(f"Output artifact {art.uuid} connected to build {art.created_by_build_id}, target {art.created_by_target_id}, step {art.created_by_step_id}")


class StorageCollection(BaseModel):
    build_storage: IStoredBuildStorage 
    artifact_registry: IArtifactRegistry
    target_storage: IStoredTargetRunStorage 
    step_storage: IStoredStepRunStorage

def _connect_artifact(build:StoredBuild, targetrun:StoredTargetRun, step:StoredStepRun, art:ArtifactRegistration, asinput:bool):
    """Attache the artifact to the step's input or outputs and connect the artifact to the build.

    Args:
        build (StoredBuild): _description_
        step (StoredStepRun): _description_
        art (ArtifactRegistration): _description_
        asinput (bool): _description_
    """
    # Connect the build to the artifact
    art.created_by_build_id = build.uuid
    art.space_name = build.space_name
    art.username = build.username

    # Connect the target and step to the artifact
    art.created_by_target_id = targetrun.uuid
    if asinput:
        index = len(targetrun.input_artifacts)
        targetrun.input_artifacts['input' + str(index)] = art.uuid
        #targetrun.input_artifact_ids.append(art.uuid)
    else:
        index = len(targetrun.output_artifacts)
        targetrun.output_artifacts['output' + str(index)] = [ art.uuid ]
        #targetrun.output_artifact_ids.append(art.uuid)
    
def _connect_build(build:StoredBuild,  targetspec: TargetSpec):
    """Link all the uuids between build, targets, step, and artifacts.

    Args:
        build (StoredBuild): _description_
        targetspec (TargetSpec): _description_
    """
    target = targetspec.target
    # Connect step to build
    target.build_id = build.uuid
    # Connect inputs to build and step
    for art in targetspec.input_artifacts:
        _connect_artifact(build, target, targetspec.step, art, True)
    # Connect outputs to build and step
    for art in targetspec.output_artifacts:
        _connect_artifact(build, target, targetspec.step, art, False)
    # Connect the steps back to the build and target
    targetspec.step.build_id = build.uuid
    targetspec.step.target_id = target.uuid

def _store_connected_build(build:StoredBuild, targets:list[TargetSpec], storage:StorageCollection):
    item = storage.build_storage.get_by_uuid(build.uuid)
    if item is not None:
        storage.build_storage.update(build)
    else:
        storage.build_storage.add(build)

    for target in targets:
        assert isinstance(target,TargetSpec)
        storage.target_storage.add(target.target)
        allarts = target.input_artifacts + target.output_artifacts
        stored_uuids = []
        for art in allarts:
            if not art.uuid in stored_uuids:    # In case artifacts are shared across steps
                storage.artifact_registry.add(art)
                stored_uuids.append(art.uuid)
        storage.step_storage.add(target.step)


def connect_and_store_build(build:StoredBuild, targets:Union[TargetSpec, list[TargetSpec]], storage:StorageCollection):
    """Connect, via uuids, the targets and steps to the build and the artifacts to the targets and build
    and store all entities.

    Args:
        build (StoredBuild): _description_
        steps (StepSpec): step and its artifacts 
        storage (Storage): collection of storage instances 
    """
    if isinstance(targets,TargetSpec):
        targets = [targets]

    for target in targets:
        # Connect the targets/steps to the build and artifacts to the target/step
        _connect_build(build, target)
        target.show_connections()

    # Store builds, steps and artifacts
    _store_connected_build(build, targets, storage)
