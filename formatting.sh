#!/bin/sh
black . -v
flake8 . --count --ignore="E501 W503" --show-source --statistics --exclude=./.*,build,dist
