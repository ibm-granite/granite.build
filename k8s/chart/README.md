# Deploy LLM.build

A Helm chart for deploying LLM.Build to Kubernetes/Openshift.

## Deploying LLM.build server on VPC namespaces

### Basic commands
Here's the instructions on a successful LLM.build server installation on the VPC namespace.

1. Basic setup
```
# Basic value setup
export LLMB_RELEASE_NAME=dev
export LLMB_NAMESPACE=llm-build-dev
cd k8s/chart
```

2. Configure the definition.
Make sure that the per-instance setting (such as `values-dev.yaml`) is correct. Specifically,
- Confirm the image name and tag. We could use the 'sed' trick to dynamically replace the image tag too.
- Determine whether to use a secret. See `secrets.mainSecretName` in `values-dev.yaml`, and more details in the later section.
- Additional preparation is needed for service account token. Again see the later section for more details.

3. Install helm

```
# Initial setup
helm install --namespace $LLMB_NAMESPACE --name-template=$LLMB_RELEASE_NAME --values values.yaml --values values-dev.yaml .
```

To list helms
```
helm ls --namespace $LLMB_NAMESPACE
```

4. Update helm

For updating the same release with the new version,
```
# For updating the same release with the new version
helm upgrade --namespace $LLMB_NAMESPACE --values values.yaml --values values-dev.yaml $LLMB_RELEASE_NAME .
```

5. Delete helm
```
# Delete the release
helm delete --namespace $LLMB_NAMESPACE $LLMB_RELEASE_NAME
```

### SPS setup

- We'll need to introduce a CI-trigger that runs the `helm update` for the given image update reusing the same release. This is easier to achieve for the short term.
- Alternatively, we can delete the existing helm with `helm delete`, and introduce a new release with `helm install` with a new release number.

## Discussions

### Instance-specific settings
helm allows multiple `--values` parameters. Using this mechanism, the instance-specific configurations are defined in a separate file, such as `values-dev.yaml`.

### Use of `.Release.Name`

The above command sets `$LLMB_RELEASE_NAME` as the name of the helm "release". In the template definitions, this value is accessed as `.Release.Name`, which we're using within various resource names.

By including `.Release.Name` in resource names, it becomes very easy to install multiple releases (i.e. server instances) in the same namespace. While this is working well in most parts, we should be careful on the following areas.
- We should be careful with naming. For example, the name like `0.2.20` doesn't work, because dots are not allowed for a service name. `0_20_20` doesn't work either as underscores aren't permitted. Alphanumerical commit hash may be ok. In the above example, I just used a string `dev` for now.
- In the current setup, the service account is also dynamically named including `.Release.Name`. While this itself works fine, the challenge is we also need to statically define a secret named `ris3-kube-config` which contains the token for this service account. See the above sections on more details.
- For the rest server route, since it's less useful to include `.Release.Name` in the host name, it's currenty set a static value such as `api.llm-build-dev.vpc-int.res.ibm.com`. This means we should be careful not to create routes from multiple releases as they will conflict with each other.

### Secret setup

By default, the template dynamically defines a secret for each release, by including `.Release.Name` in the secret name. This allows a clean and robust installation by letting helm also manage the secret definition. To make this work, however, the actual secret values need to be managed securely somewhere. Once we enable SPS based deployment, it can probably handle this part, so this will be the way to go then.
We should be careful never to push the value files containing raw secrets.

For now, the template definition optinally allows `secrets.mainSecretName` in values.yaml (or its per-env variants) to specify a static secret definition.

The additional complexity involved in the secret definition is the service account token, which is described next.

### Service account token setup

In the current setup, the secret needs to contain `ris3-kube-config`, which is used by `build-watch` to invoke `build-runner` jobs and pods. Note the following:
- This is poorly named, given that the environment is no longer RIS3. Something like `build-runner-kube-config` may be clearer.
- There's been a suggestion to define this in Secret Manager instead. This is a known TODO.
- The contents of this secret looks like the following:
```yaml
apiVersion: v1
clusters:
- cluster:
    server: https://c111-e.us-east.containers.cloud.ibm.com:30767
  name: c111-e.us-east.containers.cloud.ibm.com:30767
contexts:
- context:
    cluster: c111-e.us-east.containers.cloud.ibm.com:30767
    namespace: llm-build-dev
    user: llm-dot-build-svc-acc-{{ .Release.Name }}/c111-e.us-east.containers.cloud.ibm.com:30767
  name: llm-build-dev/c111-e.us-east.containers.cloud.ibm.com:30767/llm-dot-build-svc-acc-{{ .Release.Name }}
current-context: llm-build-dev/c111-e.us-east.containers.cloud.ibm.com:30767/llm-dot-build-svc-acc-{{ .Release.Name }}
kind: Config
preferences: {}
users:
- name: llm-dot-build-svc-acc-{{ .Release.Name }}/c111-e.us-east.containers.cloud.ibm.com:30767
  user:
    token: <token value>
```

Since the service account name is dynamic (dependent on `.Release.Name`), the content of the secret is also dependent on it.

- The token can be generated by the user with admin priviledge:
```
oc create token --duration=31536000s granite-dot-build-svc-acc-$LLMB_RELEASE_NAME --namespace $LLMB_NAMESPACE
```

Note that the the above command works only when the service account already exists.
- Putting together the above points, installation current requires the following steps. This needs to be improved.
    + Install or update helm, which makes sure that the service account and the secret exists for the given release.
    + Run the above `oc create token` command to genearate a token.
    + Craft the `ris3-kube-config` (or `build-runner-kube-config` value above, including the token and the service account name)
    + If we continue to use the OpenShift secret to store it, (1) update the secret value, and (2) restart `build-watch` to reflect it. If we move it to Secret Manager, create a secret there with proper name.
- Alternatively, we can take management of the service account out of helm, and continue to use a statically configured service account.
- Or find a way to go through all the above steps also within helm for a dynamically created service account.

### Route setup
In VPC, rounts are automatically configured to allow regular VPN access without "TUNNEL-ALL", by following the host name convention. The host name needs to follow a pattern `xxx.<VPC namespace>.vpc-int.res.ibm.com`.

Since our VPC namespaces are called `llm-build-dev`, `llm-build-staging`, and `llm-build-prod`, we don't need to repeat the instance differentiation (DEV/STAGING/PROD) in the xxx part. Therefore, we're going with the naming convention such as `api.llm-build-dev.vpc-int.res.ibm.com` going forward.

Since the above route is available for regular VPN access, we won't need a separate ingress setting. Once this is fully verified to work, we can delete the ingress from the template.

This also means that we will no longer differentiate `PROD` and `PROD_INTERNAL` in the CLI setting- we'll keep both, but they'll point to the same server URL.

We should make sure that various end-user environments can access `api.llm-build-prod.vpc-int.res.ibm.com`, port 443. 

We haven't fully verified whether the IP address won't change across helm installs. With some quick tests so far, uninstalling and reinstalling the route doesn't seem to change the IP address, so the initial observation is promising.

### Persistent volume claims setup

In the current setup, `build-watch` requies a pvc which it mounts as a large temporary storage. The helm template creates a new pvc for each release, which is a clean working solution. Note the following points:
- In RIS3, we had an issue allowing write access for the non-root pod user. To overcome this problem, we manually created a world-writable sub-folder (named `gbserver-build-watch-tmp`, with permission 777) as one-time setup. Since the helm templates don't don this, there was a concern that this will become a challenge. Fortunately, it turns out that in the new VPC namespace, without any other special setting, the pvc is wribable by the non-root pod user from `build-watch`, so no special arrangement was necessary for this part.
(Since we don't need this trick, in the future we may just mount the whole pvc instead of the subpath to simplify. Lokd at this part of the build watcher description.)
```yaml
        volumeMounts:
        - mountPath: /tmp
          name: gb-buildws-volume
          subPath: gbserver-build-watch-tmp
```
- Since the pvc is dynamically created per release, it will be deleted when the release is deleted (with `helm delete`.) It won't be deleted with `helm update` unless there's a change in pvc setting. If needed, we'll have to include additional mechanism for periodic pvc cleanups. (Since the capacity demands are much less than before, this is of less concern.)
- Since some part of pvc definition is immutable, `helm update` sometimes fails if the definition changed significantly. In such a case, we just have to manually delete the pvc and let helm start over.
- Generally, there's no need to keep the content of the pvc. It's purely a temporary storage, larger than a typical ephemeral storage capacity allows.

### Spligging templates (TBD)
Currently, there's a single set of templates for both the orchestrator instance (where `rest-server`, `build-watch` and `build-runner` run) and the computing environment (such as Vela). `values.yaml` has a flag `generateResourcesForLLMBBackendCluster` to differentiate them.

While this is fine for now, there's little overlap in what we need between them, so just splitting templates for each may further simplify the definitions.
