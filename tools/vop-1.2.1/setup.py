from distutils.core import setup, Command
from distutils.extension import Extension
from distutils.command.build_py import build_py as _build_py

from os import system
from unittest import TextTestRunner, TestLoader

class build_py(_build_py):
    system("python gen.py > generated.i")

class TestCommand(Command):
  user_options = [ ]
  def initialize_options(self):
    pass
  def finalize_options(self):
    pass
  def run(self):
    tests = TestLoader().loadTestsFromNames(['testvop'])
    t = TextTestRunner(verbosity = 1)
    t.run(tests)

setup(
  name = 'vop',
  version = '1.2.1',
  author = 'James Bowman',
  author_email = 'jamesb-vop@excamera.com',
  url = 'http://excamera.com/articles/25/vop.html',
  cmdclass = {'build_py': build_py, 'test': TestCommand},
  ext_modules = [
    Extension(
        'vop',
        ['vop.c'],
        extra_compile_args=['-msse3','-fstrict-aliasing', '-g', '-O9']
  )])
