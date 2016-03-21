"""
This are the constants used in our project.

**MIN_YEAR** = *2010*

**MAX_YEAR** = *2015*

MAX_YEAR and MIN_YEAR should be changed according to current year (up to August it should be last year) and
MAX_YEAR - 5 , due to inforamtion that is not saved before.

**YEARS** = *[str(i) for i in range(MIN_YEAR,MAX_YEAR)]*

**LEAGUES** = *['Primer_League','Serie_A','Ligue1','La_Liga','Bundesliga']*

We can Add new leagues or remove current ones.

**LEAGUES_ABV** = *['PL','SA','L1','LL','BL']*

**MONTHS** = *['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']*

"""
import os


MIN_YEAR = 2010
MAX_YEAR = 2015

YEARS = [str(i) for i in range(MIN_YEAR,MAX_YEAR)]

LEAGUES = ['Primer_League','Serie_A','Ligue1','La_Liga','Bundesliga']

LEAGUES_ABV = ['PL','SA','L1','LL','BL']

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

