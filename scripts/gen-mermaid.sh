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

# Requires types to be listed in types/__init__.py
# pip install pydantic-2-mermaid 
# From above the src directory
export PYTHONPATH=src
#pydantic-mermaid -m gbserver.types -o tt.md  -e both
#pydantic-mermaid -m gbserver.types -o tt.md  -e inheritance 
pydantic-mermaid -m gbserver.types -o tt.md  -e dependency 
# Remove '= {}' which seems to cause the previewer a problem.
cat tt.md | sed -e 's/=[ 	]*{[ 	]*}//g' > types.md
