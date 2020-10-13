default: build
	GITHUB_WORKSPACE=/scikit-fem docker run -e GITHUB_WORKSPACE -v ${PWD}:"/scikit-fem" skfem

build:
	docker build -t skfem https://github.com/kinnala/scikit-fem-docker-action.git#main

rebuild:
	docker build --no-cache -t skfem https://github.com/kinnala/scikit-fem-docker-action.git#main

run:
	docker run -it -v ${PWD}:"/scikit-fem" --entrypoint /bin/bash skfem

sphinx:
	-rm -r ../scikit-fem-docs/.doctrees/
	@eval sphinx-build -W -a -b html docs docs/_build
