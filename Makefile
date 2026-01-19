default:
	@echo "doctest      Run Sphinx doctests"
	@echo "tests        Run pytest, flake8 and doctests"
	@echo "docs         Build Sphinx documentation"
	@echo "release      Push new version to PyPI"

# Tests

tests: pytest flake8 doctest

pytest:
	pytest

flake8:
	flake8 skfem

doctest:
	@eval sphinx-build -a -b doctest docs docs/_build

# Documentation

docs:
	@eval sphinx-build -W -a -b html docs docs/_build

# Release

release:
	@echo "Steps for release:"
	@echo "1. Update changelog in README.md and modify version number in skfem/__init__.py"
	@echo "2. Create a commit, e.g., 'Bump up version number'"
	@echo "3. Push (and wait for CI to run successfully to minimize bugs)"
	@echo "4. Run 'make release'"
	@echo "5. Go to GitHub and draft a release; add tag during release"
	@echo "6. After 15 mins, go to kinnala/scikit-fem-release-tests and add new version"
	@read -p "... Press enter to build and upload the current branch ..."
	-rm -r dist
	-rm -r build
	-rm -r scikit_fem.egg-info
	flit publish --no-use-vcs

.PHONY: docs
