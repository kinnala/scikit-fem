default: build
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem

build:
	docker build -t skfem https://github.com/kinnala/scikit-fem-docker-action.git#main

sphinx:
	-rm -r ../scikit-fem-docs/.doctrees/
	@eval sphinx-build -a -b html docs ../scikit-fem-docs
