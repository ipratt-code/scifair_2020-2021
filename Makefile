run: install_prereq
	python3 $(CURDIR)/the-actual-code/disease_model.py

install_prereq: $(CURDIR)/requirements.txt
	curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
	python3 get-pip.py
	python3 -mpip install -r $(CURDIR)/requirements.txt --force-reinstall --no-cache-dir
