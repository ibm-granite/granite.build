SHELL=/bin/bash
# Useful commands for building and running
PYTHON=python3
PIP=$(PYTHON) -m pip
DOCKER ?= podman
GBSERVER_GH_DOMAIN ?= github.ibm.com
VENVDIR=.venv
VENV_INSTALL_TARGET='.[dev]'
ARTIFACTORY_DESTINATION=https://na.artifactory.swg-devops.com/artifactory/api/pypi/res-data-engineering-team-pypi-local
# Eventually this number (a percentage over all the code) needs to be bumped up.
MIN_COVERAGE?=20

GB_ENVIRONMENT_LOWER ?= dev

GIT_COMMIT ?= $(shell git rev-parse HEAD)
GIT_DIRTY  = $(shell test -n "`git status --porcelain`" && echo "dirty-" || echo "")
IMAGE_NAME = gbserver
IMAGE_TAG  ?= ${GIT_DIRTY}commit-${GIT_COMMIT}
IMAGE_NAME_AND_TAG = ${IMAGE_NAME}:${IMAGE_TAG}

OC_LOGIN_SERVER_URI ?= https://c100-e.us-south.containers.cloud.ibm.com:30049

SIDECAR_IMAGE_NAME = gb-sidecar-monitoring
# SIDECAR_IMAGE_TAG ?= 0.4.12
SIDECAR_IMAGE_TAG ?= ${GIT_DIRTY}commit-${GIT_COMMIT}
SIDECAR_IMAGE_NAME_AND_TAG = ${SIDECAR_IMAGE_NAME}:${SIDECAR_IMAGE_TAG}

DEV_IMAGE_REGISTRY = us.icr.io/cil15-shared-registry/gb-dev
DEV_IMAGE = ${DEV_IMAGE_REGISTRY}/${IMAGE_NAME_AND_TAG}
DEV_SIDECAR_IMAGE = ${DEV_IMAGE_REGISTRY}/${SIDECAR_IMAGE_NAME_AND_TAG}

STAGING_IMAGE_REGISTRY = us.icr.io/cil15-shared-registry/gb-staging
STAGING_IMAGE = ${STAGING_IMAGE_REGISTRY}/${IMAGE_NAME_AND_TAG}
STAGING_SIDECAR_IMAGE = ${STAGING_IMAGE_REGISTRY}/${SIDECAR_IMAGE_NAME_AND_TAG}

PROD_IMAGE_REGISTRY = us.icr.io/cil15-shared-registry/gb-prod
PROD_IMAGE = ${PROD_IMAGE_REGISTRY}/${IMAGE_NAME_AND_TAG}
PROD_IMAGE_TAG_LATEST = ${PROD_IMAGE_REGISTRY}/${IMAGE_NAME}:latest
PROD_SIDECAR_IMAGE = ${PROD_IMAGE_REGISTRY}/${SIDECAR_IMAGE_NAME_AND_TAG}

_SUCCESS := "\033[32m[%s]\033[0m %s\n" # Green text for "printf"
_ERROR := "\033[31m[%s]\033[0m %s\n" # Red text for "printf"

# Avoid asking the user for confirmation if we export ASK_USER_TO_CONFIRM=false
# This is meant for non-interactive usage in scripts.
ASK_USER_TO_CONFIRM ?= true
EXTRA_BUILD_PARAM ?=
PYTEST_NUM_TEST_PROC ?= auto
#PYTEST_DIST_MODE ?= worksteal 	# 26 min, but faile test_build_watcher_c/gpu
PYTEST_DIST_MODE ?= loadgroup
DEFAULT_PYTEST_MARKERS ?= not secret_manager and not nats_server and not docker_required
PR_PYTEST_MARKERS ?= $(DEFAULT_PYTEST_MARKERS) 
MERGE_PYTEST_MARKERS ?=  $(DEFAULT_PYTEST_MARKERS)
STANDALONE_PYTEST_MARKERS ?= not secret_manager and not nats_server and not docker_required and not ibm and not nats


.PHONY: ask-user-to-confirm
ask-user-to-confirm:
	@if [[ "${ASK_USER_TO_CONFIRM}" == "true" ]] ; then \
		echo -n "Are you sure? [y/N] " && read ans && if [ $${ans:-'N'} = 'y' ]; then \
			printf $(_SUCCESS) "OK" "Continuing" ; \
			exit 0 ; \
		else \
			printf $(_ERROR) "NOT OK" "Stopping" ; \
			exit 1 ; \
		fi \
	else \
		echo 'asking user for confirmation is disabled' ; \
	fi

.PHONY: info
info:
	@echo "GB_ENVIRONMENT_LOWER  : ${GB_ENVIRONMENT_LOWER}"
	@echo "DOCKER                : ${DOCKER}"
	@echo "GIT_COMMIT            : ${GIT_COMMIT}"
	@echo "GIT_DIRTY             : ${GIT_DIRTY}"
	@echo "IMAGE_NAME_AND_TAG    : ${IMAGE_NAME_AND_TAG}"
	@echo "DEV_IMAGE             : ${DEV_IMAGE}"
	@echo "STAGING_IMAGE         : ${STAGING_IMAGE}"
	@echo "PROD_IMAGE            : ${PROD_IMAGE}"
	@echo "DEV_SIDECAR_IMAGE     : ${DEV_SIDECAR_IMAGE}"
	@echo "STAGING_SIDECAR_IMAGE : ${STAGING_SIDECAR_IMAGE}"
	@echo "PROD_SIDECAR_IMAGE    : ${PROD_SIDECAR_IMAGE}"
	@echo "export GBSERVER_IMAGE_TAG=${IMAGE_TAG}"
	@echo "export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG}"
# Start the container daemon if not already running (macOS).
# Behaviour depends on DOCKER (default: podman):
#   podman  — podman machine start
#   colima  — colima start
#   docker  — open Docker Desktop
.PHONY: start-docker
start-docker:
	@if [ "${DOCKER}" = "podman" ]; then \
		echo "Starting Podman machine..."; \
		if ! podman machine list --format "{{.Running}}" 2>/dev/null | grep -q "true"; then \
			podman machine start; \
		else \
			echo "Podman machine is already running."; \
		fi; \
	elif [ "${DOCKER}" = "colima" ]; then \
		echo "Starting colima (Docker daemon)..."; \
		if ! colima status > /dev/null 2>&1; then \
			colima start; \
		else \
			echo "colima is already running."; \
		fi; \
	elif [ "${DOCKER}" = "docker" ]; then \
		echo "Starting Docker Desktop..."; \
		if ! docker info > /dev/null 2>&1; then \
			open -a Docker; \
			echo "Waiting for Docker Desktop to be ready..."; \
			until docker info > /dev/null 2>&1; do sleep 2; done; \
			echo "Docker Desktop is ready."; \
		else \
			echo "Docker Desktop is already running."; \
		fi; \
	else \
		echo "Unknown DOCKER value '${DOCKER}'. Expected: podman, colima, or docker."; \
		exit 1; \
	fi

.PHONY: check-git-status-clean
check-git-status-clean:
	@if [[ -z "${ALLOW_DIRTY}" ]] ; then if [[ -n "${GIT_DIRTY}" ]] ; then echo 'Please make sure the git status is clean or set ALLOW_DIRTY.' && exit 1 ; fi fi

.PHONY: check-container-registry-login
check-container-registry-login:
	@echo "Checking if you are logged into ${DEV_IMAGE_REGISTRY}, ${STAGING_IMAGE_REGISTRY} and ${PROD_IMAGE_REGISTRY}"
	@if [ -z "${IBM_CLOUD_API_KEY}" ]; then																\
		echo "You can login to IBM Cloud using 'ibmcloud login --sso'";									\
		echo "set the container registry region 'ibmcloud cr region-set global'";						\
		echo "and then login to the container registry with 'ibmcloud cr login --client ${DOCKER}'";	\
		echo "Or set the IBM_CLOUD_API_KEY env var for cil15 account and run this again";				\
	else														\
		echo "Logging in to IBMCloud using API key";			\
		ibmcloud login -q --apikey ${IBM_CLOUD_API_KEY}; 		\
		ibmcloud target -g granite-build;					\
		ibmcloud cr region-set us-south;							\
		ibmcloud cr login --client ${DOCKER};					\
		ibmcloud cr region-set us-south;							\
	fi


# Find the most recent commit-tagged image for either gb-staging, gb-dev or gb-prod environments
# We ignore the "latest" tagged images since sidecar does not have that tag.
# Pass in GB_ENV = one of gb_staging or gb_dev
.PHONY: .get-commit-env-help 
.get-commit-env-help: 
	@$(MAKE) check-container-registry-login
	@echo Querying the IBM Container registry.  This may take a minute or so.
	@tmpfile=/tmp/get-commit-$$;						\
	ibmcloud cr images --no-va | grep $(GB_ENV) | grep $(IMAGE_NAME) > $$tmpfile; 	\
	for time in minute hour day month; do					\
	    timez=$$(cat $$tmpfile | grep -i $$time );				\
	    if [ ! -z "$$timez" ]; then						\
		echo "Image(s) found within the last N $${time}s...";		\
	 	cat $$tmpfile | grep -i $$time |  sort -n -k 5;			\
	 	image=$$(cat $$tmpfile | grep -i $$time |  sort -n -k 5 | grep -v latest | head -n 1);		\
	 	image_tag=$$(echo $$image | awk '{print $$2}');			\
		break;								\
	    fi;									\
	done;									\
	rm $$tmpfile;								\
	echo "export GBSERVER_IMAGE_TAG=$${image_tag}";				\
	echo "export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=$${image_tag}"
	
.PHONY: get-latest-staging-image-tag
get-latest-staging-image-tag:
	$(MAKE) GB_ENV=gb-staging .get-commit-env-help

.PHONY: get-latest-dev-image-tag
get-latest-dev-image-tag:
	$(MAKE) GB_ENV=gb-dev .get-commit-env-help

.PHONY: get-latest-prod-image-tag
get-latest-prod-image-tag:
	$(MAKE) GB_ENV=gb-prod .get-commit-env-help


.PHONY: check-github-token
check-github-token:
	@if [[ -z "${GITHUB_TOKEN}" ]] ; then echo 'Please set the environment variable GITHUB_TOKEN' && exit 1 ; fi

.PHONY: check-table-prefix
check-table-prefix:
	@if [[ -z "${MY_TABLE_PREFIX}" ]] ; then echo 'Please set the environment variable MY_TABLE_PREFIX' && exit 1 ; fi

.PHONY: format
format:
	isort . && black .
	# workaround for the build directory
	isort src/gbserver/build/*
	black src/gbserver/build/*

.PHONY: staticcheck
staticcheck:
	pylint src/gbserver/ || echo '\npylint failed but run the next check anyway\n'
	mypy --disable-error-code=import-untyped src/gbserver/

.PHONY: xcheck
xcheck:
	./scripts/limited-typecheck.sh

.PHONY: xformat
xformat:
	./scripts/limited-format.sh

# Kept for backwards compatibility (for a while, since 10/3/2025)
.PHONY: cicd-test
cicd-test: 
	$(MAKE) cicd-pr-test 

.PHONY: cicd-pr-test
cicd-pr-test: 
	$(MAKE) cicd-venv
	$(MAKE) test-pr 

.PHONY: test-pr 
test-pr: 
	source $(VENVDIR)/bin/activate && \
		export GBTEST_ENABLE_EXTENDED_TESTS=false &&	\
		export GBSERVER_IMAGE_TAG=${IMAGE_TAG} && \
		export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG} && \
		args=(--durations=20 --cov --cov-report=xml --junitxml=report.xml) && \
		args+=(-n ${PYTEST_NUM_TEST_PROC} --dist=${PYTEST_DIST_MODE} -s) && \
		args+=(-m '$(PR_PYTEST_MARKERS)' --strict-markers) && \
		pytest "$${args[@]}" test/unit test/e2e test/integration/ibm && \
		coverage report --fail-under=$(MIN_COVERAGE) --sort=Cover

.PHONY: cicd-merge-test
cicd-merge-test:
	$(MAKE) cicd-venv
	$(MAKE) test-merge 

.PHONY: test-merge 
test-merge:
	#source $(VENVDIR)/bin/activate && pytest --cov -s test
	#source $(VENVDIR)/bin/activate && $(MAKE) start-docker	# Needed by some build integration tests
	source $(VENVDIR)/bin/activate && \
		export GBTEST_MODE=live && \
		export GBTEST_ENABLE_EXTENDED_TESTS=true &&	\
		export GBSERVER_IMAGE_TAG=${IMAGE_TAG} && \
		export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG} && \
		args=(--durations=20 --cov --cov-report=xml --junitxml=report.xml) && \
		args+=(-n ${PYTEST_NUM_TEST_PROC} --dist=${PYTEST_DIST_MODE} -s) && \
		args+=(-m '$(MERGE_PYTEST_MARKERS)' --strict-markers) && \
		pytest "$${args[@]}" test/unit test/e2e test/integration/ibm && \
		coverage report --fail-under=$(MIN_COVERAGE) --sort=Cover

.PHONY: py-test
py-test:
	# - Default (all tests): make py-test
	# - Specific test: make py-test ARGS=test/integration/ibm/utils/test_user_spaces_list.py
	# - Multiple args: make py-test ARGS="test/integration/ibm/api -k test_artifact_get"
	# $(MAKE) cicd-venv
	source $(VENVDIR)/bin/activate && \
		export GBTEST_ENABLE_EXTENDED_TESTS=false &&	\
		export GBSERVER_IMAGE_TAG=${IMAGE_TAG} && \
		export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG} && \
		pytest -s -m "$(DEFAULT_PYTEST_MARKERS)" --strict-markers $(or $(ARGS),test)

# Add new g4os test files to this list
G4OS_TEST_FILES = \
	test/unit/api/test_auth.py \
	test/unit/builtins/steps/test_s3pull.py \
	test/unit/builtins/steps/test_s3push.py \
	test/unit/commands/test_standalone_command.py \
	test/integration/ibm/environment/test_gpu_types.py \
	test/integration/ibm/environment/test_runpod.py \
	test/integration/ibm/environment/test_skypilot.py \
	test/integration/ibm/messaging/test_nats_messaging.py \
	test/integration/ibm/messaging/test_optional_rabbitmq.py \
	test/integration/ibm/spacesecretmanager/test_envspacesecretmanager.py \
	test/integration/ibm/spacesecretmanager/test_hybridspacesecretmanager.py \
	test/unit/standalone/test_optional_imports.py \
	test/unit/standalone/test_optional_lakehouse.py \
	test/unit/standalone/test_regression_smoke.py \
	test/e2e/standalone/test_standalone_e2e.py \
	test/e2e/standalone/test_standalone_rest_api_e2e.py

.PHONY: test-g4os
test-g4os:
	pytest -m g4os $(G4OS_TEST_FILES)

.PHONY: cicd-skypilot-pr
cicd-skypilot-pr: test-g4os

# --- Local infrastructure (SLURM + MinIO) ---

.PHONY: slurm-setup
slurm-setup:
	bash scripts/slurm/setup-slurm.sh

.PHONY: slurm-teardown
slurm-teardown:
	bash scripts/slurm/teardown-slurm.sh

.PHONY: minio-setup
minio-setup:
	bash scripts/minio/setup-minio.sh

.PHONY: minio-teardown
minio-teardown:
	bash scripts/minio/teardown-minio.sh

.PHONY: integration-test
integration-test:
	source $(VENVDIR)/bin/activate && \
		pytest -s -m skypilot_integration --strict-markers test

# --- Mock/Live test mode targets ---

.PHONY: test-mock
test-mock:
	source $(VENVDIR)/bin/activate && \
		GBTEST_MODE=mock pytest -s -m "$(DEFAULT_PYTEST_MARKERS)" --strict-markers test

.PHONY: test-live
test-live:
	source $(VENVDIR)/bin/activate && \
		GBTEST_MODE=live pytest -s -m "$(DEFAULT_PYTEST_MARKERS)" --strict-markers test/unit test/integration/ibm

.PHONY: test-live-storage
test-live-storage:
	source $(VENVDIR)/bin/activate && \
		GBTEST_MODE=mock GBTEST_LIVE_STORAGE=true pytest -s -m "$(DEFAULT_PYTEST_MARKERS)" --strict-markers test

# --- Open-source CI targets (no IBM credentials required) ---

.PHONY: test-standalone
test-standalone:
	source $(VENVDIR)/bin/activate && pytest -s -m "$(STANDALONE_PYTEST_MARKERS)" --strict-markers test/unit

.PHONY: test-docker
test-docker:
	source $(VENVDIR)/bin/activate && pytest -s test/ -m docker_required

.PHONY: test-all
test-all:
	source $(VENVDIR)/bin/activate && pytest -s test/

.PHONY: lint
lint:
	source $(VENVDIR)/bin/activate && \
		isort --check . && \
		black --check . && \
		pylint src/gbserver/ && \
		pylint src/gbcli/ && \
		pylint src/gbcommon/ && \
		mypy --disable-error-code=import-untyped src/gbserver/ && \
		mypy --disable-error-code=import-untyped src/gbcli/ && \
		mypy --disable-error-code=import-untyped src/gbcommon/

.PHONY: test-local-build
test-local-build: .check-test-env
	cd samples/tests/local_hello_world_full/ && gbserver build run && cd -

.PHONY: test-remote-build
test-remote-build: .check-test-env
	cd samples/tests/hello-gb-vela/ && gbserver build run && cd -

.PHONY: start-gitops
start-gitops: check-github-token check-table-prefix
	gbserver \
	--gb-admin-table-prefix ${MY_TABLE_PREFIX} \
	pr-watch \
	--gh-token ${GITHUB_TOKEN} \
	--config samples/config/pr-watcher-config.yaml

.PHONY: start-watching-builds
start-watching-builds: check-github-token check-table-prefix
	gbserver \
	--gb-admin-table-prefix ${MY_TABLE_PREFIX} \
	build-watch \
	--gh-token ${GITHUB_TOKEN} \
	--config samples/config/build-watcher-config.yaml

.PHONY: create-spaces
create-spaces: check-table-prefix
	gbserver \
	--gb-admin-table-prefix "${MY_TABLE_PREFIX}" \
	create-spaces \
	--spaces-path samples/config/create-staging-spaces.yaml

.PHONY: delete-tables
delete-tables: check-table-prefix
	@echo "deleting tables with the prefix: ${MY_TABLE_PREFIX} in namespace granite_dot_build.admin"
	if [[ -n "${MY_DELETE_SPACES_TABLE}" ]] ; then dmf table delete -n granite_dot_build.admin -t ${MY_TABLE_PREFIX}gb_spaces ; fi
	dmf table delete -n granite_dot_build.admin -t ${MY_TABLE_PREFIX}gb_builds
	dmf table delete -n granite_dot_build.admin -t ${MY_TABLE_PREFIX}gb_targets
	dmf table delete -n granite_dot_build.admin -t ${MY_TABLE_PREFIX}gb_steps
	dmf table delete -n granite_dot_build.admin -t ${MY_TABLE_PREFIX}gb_artifacts

.PHONY: reset-and-start-gitops
reset-and-start-gitops:
	$(MAKE) delete-tables
	$(MAKE) create-spaces
	$(MAKE) start-gitops

clean::
	@# Help: Clean up the distribution build and the venv 
	rm -rf $(VENVDIR) dist build 
	rm -rf src/*egg-info

.check-build-env:
	@if [ -z "$(ARTIFACTORY_USER)" ] || [ -z "$(ARTIFACTORY_API_KEY)" ]; then echo "You must set ARTIFACTORY_USER and ARTIFACTORY_API_KEY env vars"; false ; fi
	@echo "Artifactory checks passed."

.check-test-env: .check-lakehouse-env .check-oc-login check-github-token

.check-lakehouse-env:
	@if [ -z "$(LAKEHOUSE_ENVIRONMENT)" ] || [ -z "$(LAKEHOUSE_TOKEN)" ]; then echo "You must set LAKEHOUSE_ENVIRONMENT and LAKEHOUSE_TOKEN env vars"; false ; fi
	@echo "Lakehouse checks passed."

.PHONE: .check-oc-login
.check-oc-login:
	@oc get pods > /dev/null 2>&1;	\
	if [ $$? -ne 0 ]; then		\
	    echo "You are not logged into OpenShift.  Use oc login command";\
	    exit 1;			\
	fi


.PHONY: build 
build: $(VENVDIR) 
	source $(VENVDIR)/bin/activate;	\
	${PIP} install --upgrade build;	\
	${PYTHON} -m build

.PHONY: publish 
publish:: .check-build-env
	source $(VENVDIR)/bin/activate;	\
	twine upload --verbose --non-interactive --skip-existing \
		--repository-url ${ARTIFACTORY_DESTINATION}	\
		-u ${ARTIFACTORY_USER} \
		-p ${ARTIFACTORY_API_KEY} \
		dist/*

.PHONY: venv
venv: 	dev-venv

# This publishes a bash image for use in build tests in CI/CD 
publish-test-images-icr:
	$(DOCKER) pull bash:latest
	$(MAKE) DOCKER_IMAGE_NAME=bash DOCKER_IMAGE_VERSION=latest publish-icr

publish-icr:
	#ibmcloud login -q -u "$(IBM_CLOUD_USER)" -apikey "$(IBM_CLOUD_API_KEY)" 
	$(MAKE) ibmcloud-cr-login
	#ibmcloud cr login --client $(DOCKER) 
	$(DOCKER) tag $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION) us.icr.io/cil15-shared-registry/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)
	$(DOCKER) push us.icr.io/cil15-shared-registry/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)
	# ibmcloud cr image-list | grep $(DOCKER_IMAGE_NAME)

.PHONY: dev-venv
dev-venv:
	rm -rf $(VENVDIR)
	$(MAKE) VENV_INSTALL_TARGET='.[all,dev]' $(VENVDIR)

.PHONY: cicd-venv
cicd-venv:
	rm -rf $(VENVDIR)
	$(MAKE) VENV_INSTALL_TARGET='.[all,dev]' $(VENVDIR)	# [all,dev] installs all optional deps + test tools

.PHONY: standalone-venv
standalone-venv:
	rm -rf $(VENVDIR)
	$(MAKE) VENV_INSTALL_TARGET='.[standalone,dev]' $(VENVDIR)

.PHONY: demo-venv
demo-venv:
	rm -rf $(VENVDIR)
	$(MAKE) VENV_INSTALL_TARGET='.[standalone,docker,dev]' $(VENVDIR)

.PHONY: g4os-skypilot-venv
g4os-skypilot-venv:
	@# Help: Create a venv with standalone + skypilot + dev deps (no Artifactory required)
	rm -rf $(VENVDIR)
	$(PYTHON) -m venv $(VENVDIR)
	source $(VENVDIR)/bin/activate; \
	${PIP} install --upgrade pip; \
	${PIP} install -e '.[standalone,skypilot,dev]'

$(VENVDIR): pyproject.toml 
	$(MAKE) .check-build-env
	rm -rf $(VENVDIR) 
	$(PYTHON) -m venv $(VENVDIR) 
	echo '[global]' > $(VENVDIR)/pip.conf
	#echo "extra-index-url = https://$$ARTIFACTORY_USER:$$ARTIFACTORY_API_KEY@na.artifactory.swg-devops.com/artifactory/api/pypi/res-data-engineering-team-pypi-local/simple" >> $(VENVDIR)/pip.conf
	echo "extra-index-url = https://$$ARTIFACTORY_USER:$$ARTIFACTORY_API_KEY@na.artifactory.swg-devops.com/artifactory/api/pypi/res-data-model-factory-team-pypi-local/simple" >> $(VENVDIR)/pip.conf
	source $(VENVDIR)/bin/activate; 	\
  	${PIP} install --upgrade pip;		\
	${PIP} install -e '$(VENV_INSTALL_TARGET)';	\
	#${PIP} install pytest pytest-cov pytest-asyncio pytest-xdist coverage # Why are these not in the [dev] part of pyproject.toml

#	@if [ -z "$(GBSERVER_GITHUB_TOKEN)" ]; then echo "You must set GBSERVER_GITHUB_TOKEN env vars"; false ; fi
test-1step: .check-test-env $(VENVDIR)
	oc project granite-build-staging	# Assume RIS3 cluster and 1step test running buildrunner_type="job"
	source $(VENVDIR)/bin/activate; 	\
	pytest -s test/gbserver_test/buildwatcher/test_local_build_1step.py::TestLocalBuild1Step::test_watcher_single_build

gbcli:
	source $(VENVDIR)/bin/activate; 	\
	${PIP} install "gbcli @ git+ssh://git@${GBSERVER_GH_DOMAIN}/granite-dot-build/gbcli.git"

.PHONY: image
image: .check-build-env check-git-status-clean
	@echo "Using ${DOCKER} to build the image"
	${DOCKER} build . \
	-t ${IMAGE_NAME_AND_TAG} \
	--build-arg GBSERVER_GIT_COMMIT=${GIT_COMMIT} \
	--build-arg GBSERVER_IMAGE_TAG=${IMAGE_TAG} \
	--build-arg GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG} \
	--build-arg ARTIFACTORY_USER=${ARTIFACTORY_USER} \
	--build-arg ARTIFACTORY_API_KEY=${ARTIFACTORY_API_KEY} ${EXTRA_BUILD_PARAM}
    

.PHONY: imagex
imagex: .check-build-env check-git-status-clean
	# Cross-platform build. Allow a build on e.g. Mac M1/M2/M3 for the RIS3 cluster
	@echo "Using ${DOCKER} to build the cross-platform image"
	${DOCKER} buildx build . \
	-t ${IMAGE_NAME_AND_TAG} \
	--build-arg GBSERVER_GIT_COMMIT=${GIT_COMMIT} \
	--build-arg GBSERVER_IMAGE_TAG=${IMAGE_TAG} \
	--build-arg GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${SIDECAR_IMAGE_TAG} \
	--build-arg ARTIFACTORY_USER=${ARTIFACTORY_USER} \
	--build-arg ARTIFACTORY_API_KEY=${ARTIFACTORY_API_KEY} \
	--platform linux/x86_64 \
	--load

# This is the SPS entry point for dev
.PHONY: imagepush-dev
imagepush-dev: gbserver-imagepush-dev sidecar-imagepush-dev
# This is the SPS entry point for staging (default)
.PHONY: imagepush-staging
imagepush-staging: gbserver-imagepush-staging sidecar-imagepush-staging
# This is the SPS entry point for prod
.PHONY: imagepush-prod
imagepush-prod: gbserver-imagepush-prod sidecar-imagepush-prod

gbserver-imagepush-dev: imagex check-container-registry-login
	@echo "Tagging the dev image: ${IMAGE_NAME_AND_TAG} ${DEV_IMAGE}"
	${DOCKER} tag ${IMAGE_NAME_AND_TAG} ${DEV_IMAGE}
	@echo "Pushing the dev image ${DEV_IMAGE}"
	${DOCKER} push ${DEV_IMAGE}

gbserver-imagepush-staging: imagex check-container-registry-login
	@echo "Tagging the staging image: ${IMAGE_NAME_AND_TAG} ${STAGING_IMAGE}"
	${DOCKER} tag ${IMAGE_NAME_AND_TAG} ${STAGING_IMAGE}
	@echo "Pushing the staging image ${STAGING_IMAGE}"
	${DOCKER} push ${STAGING_IMAGE}

gbserver-imagepush-prod: imagex check-container-registry-login
	@echo "Tagging the prod image: ${IMAGE_NAME_AND_TAG} ${PROD_IMAGE}"
	${DOCKER} tag ${IMAGE_NAME_AND_TAG} ${PROD_IMAGE}
	@echo "Pushing the prod image ${PROD_IMAGE}"
	${DOCKER} push ${PROD_IMAGE}
	@echo "Tagging the prod image as latest: ${IMAGE_NAME_AND_TAG} ${PROD_IMAGE_TAG_LATEST}"
	${DOCKER} tag ${IMAGE_NAME_AND_TAG} ${PROD_IMAGE_TAG_LATEST}
	# TODO: should we push the untagged image as latest?
	# @echo "Pushing the prod image as latest ${PROD_IMAGE_TAG_LATEST}"
	# ${DOCKER} push ${PROD_IMAGE_TAG_LATEST}

deploy-rest-server:
	# This requires "Tunnel-All" VPN from outside Reserach office network
	# https://cloud.ibm.com/containers/cluster-management/clusters/bs48qfvd036s0htjca9g/overview
	# Make sure that you're logged in to the cluster- the token is short-lived
	# oc login --token=<token> --server=https://c100-e.us-south.containers.cloud.ibm.com:30049
	oc get pods -n granite-build-staging --selector='app=gbserver'
	oc delete pod -n granite-build-staging --selector='app=gbserver'

.PHONY: sidecar-image
sidecar-image: check-git-status-clean
	@echo "Using ${DOCKER} to build the sidecar image: ${SIDECAR_IMAGE_NAME_AND_TAG}"
	${DOCKER} buildx build . \
	-f Dockerfile.monitoring \
	-t ${SIDECAR_IMAGE_NAME_AND_TAG} \
	--build-arg GBSERVER_GIT_COMMIT=${GIT_COMMIT} \
	--build-arg GBSERVER_IMAGE_TAG=${IMAGE_TAG} \
	--platform linux/x86_64 \
	--load

# Deprecated
.PHONY: ibmcloud-cr-login
ibmcloud-cr-login:
	ibmcloud login -q -u "${IBM_CLOUD_USER}" -apikey "${IBM_CLOUD_API_KEY}"
	ibmcloud cr region-set us-south
	ibmcloud cr login --client ${DOCKER}

.PHONY: sidecar-imagepush-dev
sidecar-imagepush-dev: sidecar-image check-container-registry-login
	@echo "Tagging the dev sidecar image: ${SIDECAR_IMAGE_NAME_AND_TAG} ${DEV_SIDECAR_IMAGE}"
	${DOCKER} tag ${SIDECAR_IMAGE_NAME_AND_TAG} ${DEV_SIDECAR_IMAGE}
	@echo "Pushing the dev sidecar image ${DEV_SIDECAR_IMAGE}"
	${DOCKER} push ${DEV_SIDECAR_IMAGE}

.PHONY: sidecar-imagepush-staging
sidecar-imagepush-staging: sidecar-image check-container-registry-login
	@echo "Tagging the staging sidecar image: ${SIDECAR_IMAGE_NAME_AND_TAG} ${STAGING_SIDECAR_IMAGE}"
	${DOCKER} tag ${SIDECAR_IMAGE_NAME_AND_TAG} ${STAGING_SIDECAR_IMAGE}
	@echo "Pushing the staging sidecar image ${STAGING_SIDECAR_IMAGE}"
	${DOCKER} push ${STAGING_SIDECAR_IMAGE}

.PHONY: sidecar-imagepush-prod
sidecar-imagepush-prod: sidecar-image check-container-registry-login
	@echo "Tagging the prod sidecar image: ${SIDECAR_IMAGE_NAME_AND_TAG} ${PROD_SIDECAR_IMAGE}"
	${DOCKER} tag ${SIDECAR_IMAGE_NAME_AND_TAG} ${PROD_SIDECAR_IMAGE}
	@echo "Pushing the prod sidecar image ${PROD_SIDECAR_IMAGE}"
	${DOCKER} push ${PROD_SIDECAR_IMAGE}

.PHONY: openshift-login
openshift-login:
	@if [ ! -z "${OC_LOGIN_API_KEY}" ]; then \
		set -x; \
		oc login -u apikey -p "${OC_LOGIN_API_KEY}" --server "${OC_LOGIN_SERVER_URI}"; \
	else \
		echo "We assume that you are already logged in"; \
	fi

.PHONY: update-deployment
update-deployment: openshift-login
	@cat k8s/${GB_ENVIRONMENT_LOWER}/dep-gbserver-rest-server.yaml | sed "s/gbserver:latest/gbserver:commit-${GIT_COMMIT}/"
	@echo
	@echo '-------------------'
	$(eval RIS3_K8S_NAMESPACE := granite-build-${GB_ENVIRONMENT_LOWER})
	@echo "Deploying to ${RIS3_K8S_NAMESPACE}!"
	@echo 'We will replace the current deployments with yamls similar to the one above. Please check and confirm:'
	@$(MAKE) ask-user-to-confirm
	oc project ${RIS3_K8S_NAMESPACE}
	cat k8s/${GB_ENVIRONMENT_LOWER}/dep-gbserver-rest-server.yaml | sed "s/gbserver:latest/gbserver:commit-${GIT_COMMIT}/" | oc replace -f -

.PHONY: update-deployment-vpc
update-deployment-vpc:
	@cat k8s/chart/values-${GB_ENVIRONMENT_LOWER}.yaml | sed "s/gbserver:latest/gbserver:commit-${GIT_COMMIT}/"
	@echo
	@echo '-------------------'
	$(eval VPC_K8S_NAMESPACE := llm-build-${GB_ENVIRONMENT_LOWER})
	@echo "Deploying to ${VPC_K8S_NAMESPACE}!"
	@echo 'We will replace the current deployments with the above helm values. Please check and confirm:'
	@$(MAKE) ask-user-to-confirm
	oc project ${VPC_K8S_NAMESPACE}
	$(eval VPC_K8S_NAMESPACE := llm-build-${GB_ENVIRONMENT_LOWER})
	$(eval LLMB_RELEASE_NAME := ${GB_ENVIRONMENT_LOWER})
	cd k8s/chart && cat values-${GB_ENVIRONMENT_LOWER}.yaml | sed "s/gbserver:latest/gbserver:commit-${GIT_COMMIT}/" | helm upgrade --namespace ${VPC_K8S_NAMESPACE} --values values.yaml --values - ${LLMB_RELEASE_NAME} .

.PHONY: create-release-branch
create-release-branch:
	@echo Creating a release branch to be PRd into main
	scripts/create-release-branch.sh
	@echo After merging the PR to main, please use scripts/tag-main.sh to tag the release.

