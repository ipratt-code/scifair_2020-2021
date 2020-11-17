install_prereq: $(CURDIR)/requirements.txt
	curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
	python3 get-pip.py
	python3 -mpip install -r $(CURDIR)/requirements.txt

run: $(CURDIR)/the-actual-code/disease_model.py
	python3 $(CURDIR)/the-actual-code/disease_model.py