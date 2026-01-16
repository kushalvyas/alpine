"""Test script to verify optional dependencies are not loaded with core imports.

This script tests that:
1. Core `import alpine` works without optional dependencies
2. Optional subpackages fail gracefully when their deps are missing
3. Optional subpackages work when deps are installed

Run with: python tests/test_lazy_imports.py
"""

import sys
import importlib
import subprocess


def check_module_not_loaded(module_name):
    """Check if a module is NOT currently loaded in sys.modules."""
    return module_name not in sys.modules


def test_core_import_no_optional_deps():
    """Test that core alpine import doesn't load optional dependencies."""
    # Clear any cached imports
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('alpine')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import core alpine
    import alpine
    
    # Check that optional dependencies are NOT loaded
    optional_deps = ['nibabel', 'biopandas', 'open3d', 'mcubes', 'sklearn', 'pandas']
    loaded_optional = [dep for dep in optional_deps if dep in sys.modules]
    
    if loaded_optional:
        print(f"❌ FAIL: Optional deps loaded with core import: {loaded_optional}")
        return False
    else:
        print("✓ PASS: Core import does not load optional dependencies")
        return True


def test_bio_subpackage():
    """Test alpine.bio subpackage import."""
    try:
        from alpine.bio import load_nii_gz, load_pdb
        print("✓ PASS: alpine.bio imports successfully")
        
        # Verify nibabel and biopandas are now loaded
        if 'nibabel' in sys.modules and 'biopandas' in sys.modules:
            print("  ✓ nibabel and biopandas loaded as expected")
        return True
    except ImportError as e:
        print(f"⚠ SKIP: alpine.bio not available (missing deps): {e}")
        return None  # Skip, not fail


def test_mesh_subpackage():
    """Test alpine.mesh subpackage import."""
    try:
        from alpine.mesh import march_and_save
        print("✓ PASS: alpine.mesh imports successfully")
        
        # Verify open3d and mcubes are now loaded
        if 'mcubes' in sys.modules:
            print("  ✓ mcubes loaded as expected")
        return True
    except ImportError as e:
        print(f"⚠ SKIP: alpine.mesh not available (missing deps): {e}")
        return None


def test_vis_pca_subpackage():
    """Test alpine.vis.pca subpackage import."""
    try:
        from alpine.vis import pca
        print("✓ PASS: alpine.vis.pca imports successfully")
        
        # Verify sklearn is now loaded
        if 'sklearn' in sys.modules:
            print("  ✓ sklearn loaded as expected")
        return True
    except ImportError as e:
        print(f"⚠ SKIP: alpine.vis.pca not available (missing deps): {e}")
        return None


def test_grid_search_not_auto_imported():
    """Test that grid_search (which uses pandas) is not auto-imported."""
    # Clear alpine modules
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('alpine')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import alpine.utils
    from alpine import utils
    
    # grid_search should NOT be in the namespace
    if hasattr(utils, 'GridSearch'):
        print("❌ FAIL: GridSearch is auto-imported (would require pandas)")
        return False
    else:
        print("✓ PASS: GridSearch is not auto-imported from utils")
        return True


def main():
    print("=" * 60)
    print("Alpine Lazy Import Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Core import
    print("[Test 1] Core import without optional dependencies")
    results.append(test_core_import_no_optional_deps())
    print()
    
    # Test 2: grid_search isolation
    print("[Test 2] GridSearch not auto-imported")
    results.append(test_grid_search_not_auto_imported())
    print()
    
    # Test 3: Bio subpackage
    print("[Test 3] Bio subpackage")
    results.append(test_bio_subpackage())
    print()
    
    # Test 4: Mesh subpackage
    print("[Test 4] Mesh subpackage")
    results.append(test_mesh_subpackage())
    print()
    
    # Test 5: Vis/PCA subpackage
    print("[Test 5] Vis/PCA subpackage")
    results.append(test_vis_pca_subpackage())
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("❌ Some tests failed!")
        sys.exit(1)
    else:
        print("✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
