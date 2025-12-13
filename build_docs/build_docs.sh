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
cp -r ./_build/html/_static/ ../docs/_static/
cp -r ./_build/html/sources/*.html ../docs/sources/
cp -r ./_build/html/*.html ../docs/

echo "Done!"