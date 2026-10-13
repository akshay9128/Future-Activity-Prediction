"""
Test Suite: Project structure, configuration, and environment verification.
"""


from pathlib import Path
import sys


def test_python_version():
    """Verify that Python runtime is at least version 3.9."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (major == 3 and minor >= 9), f"Python 3.9+ required, found {major}.{minor}"


def test_project_directories_exist():
    """Verify that all core project directories exist."""
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        project_root / "src",
        project_root / "src" / "data",
        project_root / "src" / "models",
        project_root / "src" / "utils",
        project_root / "tests",
        project_root / "configs",
        project_root / "outputs",
    ]
    for directory in expected_dirs:
        assert directory.is_dir(), f"Expected directory does not exist: {directory}"


def test_essential_files_exist():
    """Verify that .gitignore, requirements.txt, README.md, and default_config.yaml exist."""
    project_root = Path(__file__).resolve().parent.parent
    expected_files = [
        project_root / ".gitignore",
        project_root / "requirements.txt",
        project_root / "README.md",
        project_root / "REPRODUCIBILITY.md",
        project_root / "smoke_test.py",
        project_root / "configs" / "default_config.yaml",
        project_root / "src" / "__init__.py",
    ]
    for file_path in expected_files:
        assert file_path.is_file(), f"Expected file does not exist: {file_path}"
        assert file_path.stat().st_size > 0, f"File is empty: {file_path}"


def test_gitignore_contains_critical_patterns():
    """Verify that .gitignore contains patterns to prevent committing huge binaries."""
    project_root = Path(__file__).resolve().parent.parent
    gitignore_path = project_root / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")
    critical_patterns = ["venv/", "*.mp4", "*.avi", "outputs/checkpoints/", "__pycache__/"]
    for pattern in critical_patterns:
        assert pattern in content, f"Missing critical .gitignore pattern: {pattern}"


if __name__ == "__main__":
    test_python_version()
    test_project_directories_exist()
    test_essential_files_exist()
    test_gitignore_contains_critical_patterns()
    print("All environment tests passed successfully!")


