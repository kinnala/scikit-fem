import nbformat as nbf
from pathlib import Path

install = "!pip install scikit-fem[all]"

gallery = []

for exp in Path.cwd().glob('ex*.py'):
    example = exp.read_text()
    if exp.stem in ["ex04", "ex32", "ex13", "ex41", "ex28"]:
        # require loading of meshes: does not work
        continue
    parts = example.split("\"\"\"")
    if len(parts) < 3:
        continue
    docstring = parts[1]
    code = "\"\"\"".join(parts[2:])
    gallery.append((exp.stem, docstring.split('\n')[0][2:]))
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_code_cell(install),
        nbf.v4.new_markdown_cell(docstring),
        nbf.v4.new_code_cell(code),
    ]
    with open(str(exp)[:-2] + 'ipynb', 'w') as f:
        nbf.write(nb, f)

gallery.sort(key=lambda t: int(t[0][2:]))

nb = nbf.v4.new_notebook()

intro = """# Gallery of interactive examples

The following notebooks are generated automatically from the examples in
scikit-fem [source code distribution](https://github.com/kinnala/scikit-fem/tree/master/docs/examples).

Run the first cell of the notebook to install dependencies.

"""

for stem, topic in gallery:
    intro += "- [{}](https://colab.research.google.com/github/kinnala/scikit-fem-notebooks/blob/main/{}.ipynb)\n".format(topic, stem)

nb['cells'] = [
    nbf.v4.new_markdown_cell(intro),
]
with open('intro.ipynb', 'w') as f:
    nbf.write(nb, f)
