#!/bin/bash

# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is provided by Taiga to create the release branch which we then create a PR
# back into the main branch to trigger deployment from main
# Usage: create-release-branch.sh
# Make sure your local branch is up-to-date. This is obvious.
set -x
git checkout main
git pull --ff-only
git checkout dev
git pull --ff-only
# Find out the latest commit revision
git rev-parse HEAD
export BRANCH=release-`git rev-parse HEAD`
git checkout  -b $BRANCH
# The following is optional, but consider doing it when the main branch diverted from dev (i.e. dev isn't the direct descendent of main)
#git merge -s ours main
# push the branch
git push --set-upstream origin $BRANCH
# on github, create a PR from $BRANCH to main
set +x
echo Next, please create a PR from $BRANCH into the main branch.
