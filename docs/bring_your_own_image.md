# Bring Your Own Image into LLM.build

This document describes how to bring your own image into LLM.build.

## Prerequisites
1. Install the LLM.build CLI.
2. Use LLM.build CLI to upload any artifacts that your build will be using to Data LakeHouse (if not already there).
3. Use LLM.build CLI to upload any image pull secrets to the Secret Manager.
4. Define a build.

### Install the LLM.build CLI
The instructions to install LLM.build CLI are in the [Getting started section of the LLM.build documentation](https://pages.github.ibm.com/granite-dot-build/quick-start-guide/#getting-started)
### Upload Input Artifacts to Data Lakehouse (Optional)
To upload a fileset:
```
$ llmb artifact push --type fileset --from-local sample_fileset_artifact/ --artifact-name cma_sample_artifact
```
To upload a model:
```
$ llmb artifact push --type model --from-local sample_fileset_artifact/ --model-type granite --size 2b --variant instruct --artifact-name cma_sample_model
```

To check whether a model is already uploaded:
```
$ llmb artifact list | grep model | grep granite-3.3-2b
c9045b87-c899-494b-b75a-a263a9aee1bc  granite-3.3-2b-base.r250409a                                                                                lh://staging/base_training/models/model_shared/granite-3.3-2b-base/r250409a                                                                                      model        success                                         taiga                   May 10 2025   []
135253af-579c-42fa-8287-e3987fee23d3  granite-3.3-2b-base.r250409a                                                                                lh://prod/base_training/models/model_shared/granite-3.3-2b-base/r250409a                                                                                         model        success                                         taiga                   Aug 25        []
613a262e-a0b8-4194-aa55-8b83868aed2b  granite-3.3-2b-instruct.r250409a                                                                            lh://prod/base_training/models/model_shared/granite-3.3-2b-instruct/r250409a                                                                                     model        success                                         taiga                   Aug 25        []
```

### Upload Image Pull Secret to Secret Manager (Optional)
This operation is needed to pull an image from a registry external to LLM.build. If the image is located in the `us.icr.io/cil15-shared-registry/`, then uploading a secret is not needed. This command:
```bash
gb secret create --from-file ghcr-secret.json --personal --format json cma-ghcr-secret
```
uses as input a standard Docker `config.json` secret, e.g. the `ghcr-secret.json` file has this content:
```json
{
  "auths": {
    "ghcr.io": {
      "username": "cmadam",
      "password": "ghp_...",
      "email": "cmadam@us.ibm.com",
      "auth": "Y21...VAK"
    }
  }
}
```
When the secret is not needed anymore, it can be deleted, using:
```
llmb secret delete --personal cma-ghcr-secret
```

If you are using a Mac with `docker-credential-desktop` and access your image using IBM Cloud
credentials. Assuming that your image is in `icr.io`, you can retrieve a docker-like `config.json`
file by using this command:
```bash
echo "https://icr.io"| docker-credential-desktop get
```
This will return a token (`xxxx`) that you can add to a `config.json` file like this:

```json
{
  "auths": {
    "icr.io": {
      "auth": "aWFtcmVmcmVzaDo=",
      "identitytoken": "xxxx"
    },
    "us.icr.io": {
      "auth": "aWFtcmVmcmVzaDo=",
      "identitytoken": "xxxx"
    }
  },
}
```

### Editing the Build File
Customize the `build.yaml` template below, following these steps.

List the available templates, and select the `BYOI` template:
```
(venv) cma:test_byoi_image_registry_gpu$ gb template list
LLM.build list templates
📝 Listing templates from granite-dot-build/assets.git repository:                                                                                             
TEMPLATE NAME                            DESCRIPTION                                                                                                                                                                
BYO-CustomEval                           https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYO-CustomEval
BYOI                                     https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYOI
BYOSAgentTrajectoryTuner                 https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYOSAgentTrajectoryTuner
...
```
Initiate a build from the BYOI template:
```
(venv) cma:regression_tests$ gb build init --from-template BYOI test-byoi-build
LLM.build build init
✅ Build test-byoi-build was successfully created from template BYOI in https://github.ibm.com/granite-dot-build/assets.git@gbspace-config-dev.
```
Edit the `build.yaml` file located inside the build directory (in our case `test-byoi-build`):
1. Specify the inputs for your build:
   ```
    inputs:
      input_artifact_path:
        ### USER INPUT 1: points to a fileset used as an input artifact
        uri: lh://prod/granite_dot_build.public/filesets/fileset_shared/byos-rits-public/20250407T124342
      model_to_use:
        ### USER INPUT 2: points to a model used as an input artifact
        uri: lh://prod/base_training/models/model_shared/granite-3.3-2b-instruct/r250409a
   ```
2. Specify the outputs for your build:
   ```yaml
    outputs:
      output_1:
        ### USER INPUT 3: output_1 represents the artifact id; in this example, this is a fileset
        uri: lh://{{ space.variables.DEFAULT_LH_ENVIRONMENT }}/{{ space.variables.DEFAULT_LH_NAMESPACE }}/filesets/{{ space.variables.DEFAULT_LH_FILESET_TABLE }}/byoi-output-{{ run_metadata.targetsteprun_id | short_hash }}/1/
   ```
3. Specify the image that will be used by the build:
    ```yaml
    config:
      k8s:
        ### USER INPUT 4: pointer to the custom image to use in the build
        image: ghcr.io/cmadam/hello-ghcr:0.0.24
    ```
4. If the image is located in an external registry, for which `llmb` does not have access, specify
the image pull secret created above:
    ```yaml
      # pointer to the custom image pull secrets
      secrets:
        ### USER INPUT 5: specify image pull secret if image not in us.icr.io/cil15-shared-registry
        secret_names_to_use_as_pull_secret:
          - cma-ghcr-secret
    ```
5. The `config/env/k8s` section contains two environment variables:
   * `APP_COMMAND` contains the command that will start the execution of the image. The image start
     command can reference the artifacts specified as inputs and create an output folder for the
     output artifact(s).
   * `ECHO_COMMAND` contains the log message that will be printed by the image . This log line will
     signal completion of the user process launched by the build and will trigger the `lhpush`
     operations that push the generated artifacts from the pvc to data lakehouse.
   ```yaml
      env:
        ### USER INPUT 6: - set the command that starts the app ($APP_COMMAND) and the log message signalling completion ($ECHO_COMMAND).
        ###               - make sure the LLMB_ARTIFACT_ID matches the dictionary keys specified under 'outputs'
        APP_COMMAND:
          value: "python /app/hello_world.py --fileset-location {{ bindings.input_artifact_path.binding.path }} --model-location {{ bindings.model_to_use.binding.path }} --output-path /gb-read-write/outputs/byoi/{{ run_metadata.targetsteprun_id | short_hash }} --simulate-blackbox"
        ECHO_COMMAND:
          value: "echo LLMB_ARTIFACT_ID:output_1 LLMB_ARTIFACT_PATH:/gb-read-write/outputs/byoi/{{ run_metadata.targetsteprun_id | short_hash }}"
   ```

## Running a New Build Using a Custom Image
To run a new build: install LLM.build CLI.
List the available templates, and select the `BYOI` template:
```
(venv) cma:test_byoi_image_registry_gpu$ gb template list
LLM.build list templates
📝 Listing templates from granite-dot-build/assets.git repository:                                                                                                                                                  
TEMPLATE NAME                            DESCRIPTION                                                                                                                                                                
BYO-CustomEval                           https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYO-CustomEval
BYOI                                     https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYOI
BYOSAgentTrajectoryTuner                 https://github.ibm.com/granite-dot-build/assets/tree/gbspace-config-dev/templates/BYOSAgentTrajectoryTuner
...
```
Initiate a build from the BYOI template:
```
(venv) cma:regression_tests$ gb build init --from-template BYOI test-byoi-build
LLM.build build init
✅ Build test-byoi-build was successfully created from template BYOI in https://github.ibm.com/granite-dot-build/assets.git@gbspace-config-dev.
```
Edit the `build.yaml` file located inside the build directory (in our case `test-byoi-build`), and start the build:
```
(venv) cma:test-byoi-build$ llmb build start
🏁 LLM.build build start
(1/3) Prepared build contents.                          
(2/3) Validated build contents.  
(3/3) Submitted build request.                
✅ Requested build: https://ui.dmf-staging.vpc-int.res.ibm.com/gb/builds/69778203-9851-4ef3-b8f2-c7fb192b5375 

 gb build list | grep 69778203-9851-4ef3-b8f2-c7fb192b5375
 gb build list --show-all # See all your builds, including old ones.
To get the build status:                                               
 gb build status 69778203-9851-4ef3-b8f2-c7fb192b5375         

To get the last 10k lines of the logs:                                         
 gb build log --all 69778203-9851-4ef3-b8f2-c7fb192b5375 
By default this gives you the logs of the last step in the build. To get the logs of a particular step you can use:                                          
 gb build log --all 69778203-9851-4ef3-b8f2-c7fb192b5375 --build-step-id <step id> 
```

### Debugging
When running in Kubernetes environment, reduce the amount of logging, to lessen the load on our servers, and prevent exhaustion of the ephemeral storage. Ideally, the number of log lines should be kept under 1000 lines. Build logs can be accessed using the `llmb build log` command, listed above.

### Getting the Status of the Run
Please refer to the [Getting Started Guide](https://pages.github.ibm.com/granite-dot-build/assets/getting_started_example/)
