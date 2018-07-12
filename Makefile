default:
	ipython -m unittest discover -- -v

coverage:
	coverage run -m unittest discover
	coverage report
	coverage html
