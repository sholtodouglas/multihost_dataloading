"""Install requirements for dataloading examples."""

import os
import sys
import setuptools

_jax_version = '0.2.16'
_jaxlib_version = '0.1.76'

setuptools.setup(
    name='multihost_dataloading',
    version='0.1',
    description='multihost_dataloading examples',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='TODO:replace when updated',  # TODO
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        f"jax[tpu]>={_jax_version}",
        f'jaxlib >= {_jaxlib_version}',
        'numpy',
        'seqio-nightly',
        'tensorflow-datasets',
        'tensorstore >= 0.1.20',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='data machinelearning',
)

# extras_require={
#     'gcp': [
#         'gevent', 'google-api-python-client', 'google-compute-engine',
#         'google-cloud-storage', 'oauth2client'
#     ],
# },
