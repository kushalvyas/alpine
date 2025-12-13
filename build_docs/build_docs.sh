echo "Generating API documentation..."
sphinx-apidoc -o ./sources ../alpine/

echo "Making clean..."
make clean

echo "Making html..."
make html

echo "Creating docs subdirectories..."
mkdir -p ../docs/
mkdir -p ../docs/_static/
mkdir -p ../docs/sources/

echo "Copying static files to docs..."
cp -r ./_build/html/* ../docs/

echo "Done!"