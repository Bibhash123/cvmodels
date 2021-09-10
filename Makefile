PHONY: dist

install:
	python setup.py install

clean:
	rm -rf build/ dist/ cvmodels.egg-info/ __pycache__/ */__pycache__/
	rm -f *.pyc */*.pyc

dist:
	python3 setup.py sdist bdist_wheel