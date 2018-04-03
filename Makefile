default: test_ipynb test_flat
	@echo "Done!"

test_flat:
	ls examples | grep .py$ | xargs -I {} ipython examples/{}

test_ipynb:
	ls examples | grep .ipynb | xargs -I {} jupyter nbconvert --to html --execute examples/{}
