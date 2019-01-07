#!/usr/bin/env python
from __future__ import division, print_function

import os


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('numpy', parent_package, top_path)

    # Note: this doesn't work.  See `get_config_cmd()` in other setup.py files.
    # We also need to deal with parallel compilation, so it looks like there's
    # no good time/place to do this ...
    # Alternative is to insert the flag in numpy.distutils (only when we're
    # compiling numpy itself).  However, the default C flags come from Python
    # distutils, from CCompiler._get_cc_args() and there's no good way to
    # edit them (would need more monkeypatching, and then we'd have to run the
    # check for gcc < 5.0 on every compile command.
    if config.get_config_cmd().check_compiler_gcc_c99_default_okay():
        os.environ['CFLAGS'] = '-std=c99' + ' ' + os.environ.get('CFLAGS', '')

    config.add_subpackage('compat')
    config.add_subpackage('core')
    config.add_subpackage('distutils')
    config.add_subpackage('doc')
    config.add_subpackage('f2py')
    config.add_subpackage('fft')
    config.add_subpackage('lib')
    config.add_subpackage('linalg')
    config.add_subpackage('ma')
    config.add_subpackage('matrixlib')
    config.add_subpackage('polynomial')
    config.add_subpackage('random')
    config.add_subpackage('testing')
    config.add_data_dir('doc')
    config.add_data_dir('tests')
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
