# AI_Project 236502 - Football Scores Classification

## Synopsis

An AI project dedicated to create the best possible classifier for football scores classification.
Included are the implementations for the web crawler, database handling and dataset creation, experimentation (to find the best classifier) and different utilities.  


## Installation

Under Linux distributions, change into the project root and run the following commands:

	sudo chmod +x install.sh
	sudo ./install.sh

## Basic Usage

The two main modules of the project are the web crawler module (whoscored_crawler.py) and the experiments module (experiments.py).

### Web Crawler Usage

To crawl and collect data from the German Bundesliga for the year 2010:

	sudo python whoscored_crawler.py Bundesliga 2010
	
To crawl and collect data from the German Bundesliga for the years 2010-2015:

	sudo python whoscored_crawler.py Bundesliga 2010
	
To crawl and collect data from the current year:

	sudo python whoscored_crawler.py -u
	
For more information:
	
	sudo python whoscored_crawler.py -h
	
### Experiments Usage

To run the Best Params experiment and save the results in Results/Params:

	sudo python experiments.py Params Best_Params run
	
To report the results of the Best Params experiment that saved his results in Results/Params:

	sudo python experiments.py Params Best_Params report
	
To report the results of the Final Season experiment that saved his results in Results/Final:

	sudo python experiments.py Final Final_Season report
	
To report the results of the Final Season experiment that saved his results in Results/Final with outputting prediction for final season:

	sudo python experiments.py Final Final_Season report -v 1
	
For more information:
	
	sudo python experiments.py -h
	
*As a general rule of thumb, running an experiment report with -v 1 plots graph to the screen and create other additional data, while running report with -v 0 (or without -v at all) will only print the results and tables.*

## API Reference

API reference is under documentation/build/index.html.

## Contributors

Ory Jonay and Dror Porat.