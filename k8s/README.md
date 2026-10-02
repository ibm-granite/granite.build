# Deploying LLM.build to a cluster

## Assumptions
- No CI/CD pipeline set up yet
- Image needs to be built on local machine (e.g. M1 Mac), which requires cross-platform build.
- Use the cil15 shared container registry to push the image. Any registry is fine, as long as you can also put a shared key in the target cluster to read the image. This key is visible to the users who can access the namespace.
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

## Deploying

Deployments use the Helm chart in [chart/](chart/README.md). To roll out a new image to an environment:

```
GB_ENVIRONMENT_LOWER=staging make update-deployment-vpc
```
