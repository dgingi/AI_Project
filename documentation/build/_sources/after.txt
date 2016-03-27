Setting up auto mode on Linux
******************************************************

One of the easiest thing to do with this project, after installing and looking at the results from the experiments that we've ran, is setting it up to run
each week and build a new classifier from the new games.

To do just that, all you need to to is just set a **crontab task** to run weekly. If you don't know what crontab is, or how to set up new tasks, just click
`here <https://help.ubuntu.com/community/CronHowto>`_.

After opening the crontab editor of choice, simply add those two lines *(under the assumption that the code is in /home/AI_Project)*::
	
	  0 01 * * 1 /usr/bin/python /home/AI_Project/whoscoredcrawler.py -u
	  0 01 * * 2 /usr/bin/python /home/AI_Project/experiments.py Final_Season Final_Season run
	  
And now, every Monday the crawler will download new games and on Tuesday a new classifier will be built with the new games.
	  
