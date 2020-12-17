from distutils.core import setup, Extension



setup(name="cndarray", version='1.0.1',
      ext_modules=[Extension("cndarray", ["cndarray.c"])])