#!/bin/sh
python setup.py sdist bdist_wheel
twine upload dist/*
rm -rf build dist diff_binom_confint.egg-info
