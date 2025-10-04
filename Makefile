pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	pip install build
	python3 -m build --sdist

test:
	python3 -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"

test_degradation:
	g++ -std=c++11 -O3 -march=native -DHAVE_CXX0X -fpic -ftree-vectorize tests/cpp/graph_degradation_test.cpp -o tests/cpp/graph_degradation_test
	./tests/cpp/graph_degradation_test

test_degradation_quick:
	g++ -std=c++11 -O3 tests/cpp/quick_degradation_test.cpp -o tests/cpp/quick_degradation_test
	./tests/cpp/quick_degradation_test

test_degradation_sift:
	g++ -std=c++11 -O3 tests/cpp/sift_degradation_test.cpp -o tests/cpp/sift_degradation_test
	./tests/cpp/sift_degradation_test

test_large_sift:
	g++ -std=c++11 -O3 tests/cpp/large_sift_degradation_test.cpp -o tests/cpp/large_sift_degradation_test
	./tests/cpp/large_sift_degradation_test

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so tests/cpp/graph_degradation_test tests/cpp/quick_degradation_test tests/cpp/sift_degradation_test tests/cpp/large_sift_degradation_test

.PHONY: dist
