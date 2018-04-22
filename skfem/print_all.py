import importlib
import types

class C: pass

for fname in ['mesh',
              'assembly',
              'mapping',
              'utils',
              'element',]:
    mod = importlib.import_module(fname, 'skfem')
    for x in dir(mod):
        if isinstance(getattr(mod, x), types.FunctionType):
            print(" '"+x+"',")
        elif isinstance(getattr(mod, x), type(C)):
            print(" '"+x+"',")
