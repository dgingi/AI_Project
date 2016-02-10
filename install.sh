#!/bin/bash
# Install script for dependecies- works for Linux only :(

#Installing MongoDB
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

#Installing numpy, scipy, matplotlib, 
sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
sudo apt-get install python-matplotlib

#Installing pyMongo, scikit-learn, Selenium
sudo -H pip install pymongo
sudo -H pip install selenium
sudo -H pip install sklearn
sudo -H pip install progress==1.2
