# This Makefile can be used to locally run the container-based tests run by
# Github Actions for https://github.com/kinnala/scikit-fem.

default: test_py37
	@echo "Done!"

# Python 3.6

test_py36: build_py36
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem:py36

build_py36:
	docker build -t skfem:py36 https://github.com/kinnala/scikit-fem-docker-action.git#py36

run_py36:
	docker run -it -v ${PWD}:"/scikit-fem" --entrypoint /bin/bash skfem:py36

# Python 3.7

test_py37: build_py37
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem:py37

build_py37:
	docker build -t skfem:py37 https://github.com/kinnala/scikit-fem-docker-action.git#py37

run_py37:
	docker run -it -v ${PWD}:"/scikit-fem" --entrypoint /bin/bash skfem:py37

# Python 3.8

test_py38: build_py38
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem:py38

build_py38:
	docker build -t skfem:py38 https://github.com/kinnala/scikit-fem-docker-action.git#py38

run_py38:
	docker run -it -v ${PWD}:"/scikit-fem" --entrypoint /bin/bash skfem:py38

# Python 3.9

test_py39: build_py39
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem:py39

build_py39:
	docker build -t skfem:py39 https://github.com/kinnala/scikit-fem-docker-action.git#py39

run_py39:
	docker run -it -v ${PWD}:"/scikit-fem" --entrypoint /bin/bash skfem:py39

# Documentation

sphinx:
	-rm -r ../scikit-fem-docs/.doctrees/
	@eval sphinx-build -W -a -b html docs docs/_build

sphinx_doctest:
	@eval sphinx-build -a -b doctest docs docs/_build

# Release

release:
	-rm -r dist
	-rm -r build
	-rm -r scikit_fem.egg-info
	python -m pep517.build --source --binary .
	twine upload dist/*
