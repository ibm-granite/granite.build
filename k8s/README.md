# Deploying LLM.build to a cluster (e.g. RIS3-INT-DAL12-OCP)

## Assumptions
- No CI/CD pipeline set up yet
- Image needs to be built on local machine (e.g. M1 Mac), which requires cross-platform build.
- Use the cil15 shared container registry to push the image. Any registry is fine, as long as you can also put a shared key in the RIS3-INT-DAL12-OCP cluster to read the image. This key is visible to the users who can access the namespace.
- Use an artifactory to download DMF-library instead of a GitHub repo. To use a GitHub repo as a source of `pip`, an extra arrangement is needed to set up ssh within the Dockerfile. This means, however, that there's a change made to pyproject.toml.

## Build gbserver image
You should have an artifactory credential that can connect to the DMF library folder. Set it in the environment variables as follows.

```
export ARTIFACTORY_USER=...
export ARTIFACTORY_API_KEY=...
```

The following command creates a cross-platform image.
```
make imagex
```

If a cross-platform build isn't necessary, or your local environment is already `linux/x86_64', just a regular `docker build` works too with the following command.
```
make image
```


## Upload image to a container registry
To build a cross-platform image and push it to the registry, obtain a valid IBM Cloud key that in the account which the below registry belongs to (cil15) at https://cloud.ibm.com/iam/apikeys make sure to use the key. Set it as an environment variable

```
export CLOUD_API_KEY=...
```

## Restarting the pod to reflect the change

The below method is just to kill a running pod, which makes the OpehShift deployment pulls a new image and creates a new pod.

```
make deploy-rest-server
```


## Deploying gbserver for the first time

Here the approach is a simple set of descriptors (without helm), which consists of deployment, service, and ingress.

To access the OpenShift console,
- Go to the [RIS3-INT-DAL12-OCP cluster](https://cloud.ibm.com/containers/cluster-management/clusters/bs48qfvd036s0htjca9g/overview), where you're deploying an app. Make sure you select the account RIS3 ("1844269 - RIS3") to see it.
- Click "OpenShift web console" to access the OCP console, but that requires "TUNNELALL" VPN from ourside the office network.
- From the top-right on the web console, click your account name, and "Copy login command". Copy the command with the token, that would look like:
```
oc login --token=<token> --server=https://c100-e.us-south.containers.cloud.ibm.com:xxxxx
```
- Make sure that you can see the `granite-build-staging` namespace.
```
oc get deployments -n granite-build-staging
```

Either run the command for each descriptor, or add each the on UI (by copying/pasting the YAML file contents).
```
cd k8s/staging
oc apply -f gbserver-configmap.yml
oc apply -f deployment.yml
oc apply -f service-gbserver.yml
```

