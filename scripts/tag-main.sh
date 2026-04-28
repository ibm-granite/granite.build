#!/bin/bash
# Tags the main branch with a tag provided on the command line.
# tags are generally of the form vX.Y.z, for example v0.2.36 
# Warning: there is not check for tag collision
#
tag=$1
if [ -z "$tag" ]; then
    echo tag value must be provided
    exit 1
fi
# After PR to main is merged
git checkout main
git pull --ff-only
# List existing tags
git tag
# Define a new tag
git tag $tag 
git push origin $tag 
