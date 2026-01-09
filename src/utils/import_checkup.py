"""Import validation script to check all __init__.py files are correct.

This script validates that:
1. All __init__.py files have valid Python syntax
2. All items listed in __all__ actually exist in the corresponding modules
3. No orphaned imports (importing things that don't exist)
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple


def extract_exports_from_init(init_file: Path) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Extract __all__ list and import statements from __init__.py file.

    Returns:
        Tuple of (__all__ items, list of (module, imported_names) tuples)
    """
    try:
        with open(init_file, 'r') as f:
            tree = ast.parse(f.read())

        all_items = []
        imports = []

        for node in tree.body:
            # Find __all__ definition
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            all_items = [elt.s if isinstance(elt, ast.Str) else elt.value
                                       for elt in node.value.elts
                                       if isinstance(elt, (ast.Str, ast.Constant))]

            # Find import statements
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle both explicit relative (.module) and implicit relative (module) imports
                    module_name = node.module[1:] if node.module.startswith('.') else node.module
                    imported_names = [alias.name for alias in node.names if alias.name != '*']
                    if imported_names:
                        imports.append((module_name, imported_names))

        return all_items, imports

    except Exception as e:
        print(f"  ✗ Error parsing {init_file}: {e}")
        return [], []


def extract_definitions_from_module(module_file: Path) -> Set[str]:
    """Extract function and class names defined in a Python module."""
    try:
        with open(module_file, 'r') as f:
            tree = ast.parse(f.read())

        definitions = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions.add(node.name)
            # Also capture top-level constants
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        definitions.add(target.id)

        return definitions

    except Exception as e:
        print(f"  ✗ Error parsing {module_file}: {e}")
        return set()


def validate_package(package_path: Path, package_name: str) -> bool:
    """Validate a package's __init__.py file."""
    print(f"Validating: {package_name}")
    print(f"  Path: {package_path}")

    init_file = package_path / "__init__.py"
    if not init_file.exists():
        print(f"  ✗ No __init__.py found")
        return False

    # Extract exports and imports from __init__.py
    all_items, imports = extract_exports_from_init(init_file)

    if all_items:
        print(f"  ✓ __all__ defined with {len(all_items)} items")
    else:
        print(f"  ⚠ No __all__ defined")

    # Check each import
    all_valid = True
    for module_name, imported_names in imports:
        module_file = package_path / f"{module_name}.py"

        if not module_file.exists():
            print(f"  ✗ Module file not found: {module_file}")
            all_valid = False
            continue

        # Get definitions from the module
        definitions = extract_definitions_from_module(module_file)

        # Check if all imported names exist
        missing = [name for name in imported_names if name not in definitions]

        if missing:
            print(f"  ✗ From {module_name}: Missing {missing}")
            print(f"     Available: {sorted(definitions)}")
            all_valid = False
        else:
            print(f"  ✓ From {module_name}: All {len(imported_names)} imports valid")

    # Check that all items in __all__ are actually imported/defined
    if all_items:
        imported_set = set()
        for _, names in imports:
            imported_set.update(names)

        missing_in_all = [item for item in all_items if item not in imported_set]
        if missing_in_all:
            print(f"  ✗ Items in __all__ but not imported: {missing_in_all}")
            all_valid = False

    print(f"  {'✓' if all_valid else '✗'} Validation {'passed' if all_valid else 'failed'}\n")
    return all_valid


def main():
    """Run validation for all packages."""
    print("=" * 80)
    print("IMPORT VALIDATION CHECKUP")
    print("=" * 80)
    print()

    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"

    packages = [
        (src_path, "src (root)"),
        (src_path / "losses", "src.losses"),
        (src_path / "models", "src.models"),
        (src_path / "training", "src.training"),
        (src_path / "utils", "src.utils"),
    ]

    results = []
    for package_path, package_name in packages:
        if package_path.exists():
            success = validate_package(package_path, package_name)
            results.append((package_name, success))
        else:
            print(f"Package not found: {package_path}\n")
            results.append((package_name, False))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for package_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {package_name}")

    print()
    print(f"Result: {passed}/{total} packages passed")

    if passed == total:
        print("\n✓ All imports are valid!")
        return 0
    else:
        print("\n✗ Some imports failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
