test:
	flake8
	mypy ./*.py

install: download
	pip install flake8 autopep8 mypy

download:
	mkdir res || true
	wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip -O res/gzip.zip
	unzip res/gzip.zip -d res/

clean:
	rm -rf res
