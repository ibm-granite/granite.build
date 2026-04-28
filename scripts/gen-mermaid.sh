# Requires types to be listed in types/__init__.py
# pip install pydantic-2-mermaid 
# From above the src directory
export PYTHONPATH=src
#pydantic-mermaid -m gbserver.types -o tt.md  -e both
#pydantic-mermaid -m gbserver.types -o tt.md  -e inheritance 
pydantic-mermaid -m gbserver.types -o tt.md  -e dependency 
# Remove '= {}' which seems to cause the previewer a problem.
cat tt.md | sed -e 's/=[ 	]*{[ 	]*}//g' > types.md
