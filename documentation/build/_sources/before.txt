Installation
************

Described below are the installation procedures to run the project.

The project is written in Python2.7 - please make sure to download correct packages!

We personally recommend installing an Ubuntu workstation on a virtual machine and using the VM instead of installing the project personal computers.  

Linux Installation
==================

To install under Linux, run in terminal under the project root directory::

	sudo chmod +x install.sh
	sudo ./install
	
And **voilà**! Everything is ready!

Windows / Mac OS Installation
=============================

Before you start using our project you need to install MongoDB, Python2.7, pip and additional packages.

Python2.7 and pip
-----------------

Under Windows OS, we suggest installing the Anaconda Python distribution (it will save alot of time installing different packages from PyPI).
You can download and install the distribution from `here <https://www.continuum.io/downloads>`_.  

MongoDB
-------

	#. Please enter this `link <https://docs.mongodb.org/manual/installation/#tutorials>`_ and follow the tutorial that suits your OS to install and configure MongoDB.
	
	#. Run from the project root directory::
	
		mongorestore -d leagues_db server_dump/leagues_db
	
	

Additional packages
-------------------
	Install the following packages according to your OS (should be installed if using Anaconda distribution):
	
	#. `NumPy <http://docs.scipy.org/doc/numpy-1.10.1/user/install.html>`_
	
	#. `SciPy <http://www.scipy.org/install.html>`_
	
	#. `MatplotLib <http://matplotlib.org/users/installing.html>`_
		
	Run from project root::
	
		pip install -r REQUIREMENTS.txt 
