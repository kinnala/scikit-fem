default:
	ipython -m unittest discover -- -v

coverage:
	coverage run -m unittest discover
	coverage report
	coverage html

release:
	@eval python setup.py sdist upload

dox:
	@eval sphinx-autobuild docs docs/_build/html
