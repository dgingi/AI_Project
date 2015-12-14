from Tkinter import *
from crawler import *
from old_utils import *
from pickle import load
from pickle import dump
import subprocess
import tkMessageBox
import Tkinter
import ttk
import datetime
import os
import threading
  
TITLE_FONT = ("Helvetica", 18, "bold")
BUTTON_FONT = ("Helvetica", 13, "bold")
SELECTION_FONT = ("Helvetica", 11)

class myThread (threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.ex = None
        self.ta = None
    def run(self):
        self.ex, self.ta = self.func(self.args)
        
def start_crawl_func(args,league,year):
    if league == "Nothing Selected Yet" or year == "Nothing Selected Yet":
        tkMessageBox.showinfo("ATTENTION!!!!", "Please select both league and year")
    else:
        subprocess.Popen(args+[league,year])

def get_examples(league,year,file_name):
    if league == "Nothing Selected Yet" or year == "Nothing Selected Yet" or file_name =="":
        tkMessageBox.showinfo("ATTENTION!!!!", "Please select both league,year and file name")
    else:
        if len(year.split('-')) == 1:
            if not os.path.exists(league+'-'+year+'\\'+league+'-'+year+'-May.pckl'):
                tkMessageBox.showinfo("ATTENTION!!!!", "No data for current year to get examples")
                return
        else:
            for rel_year in range(int(year.split('-')[0]),int(year.split('-')[1])):
                if not os.path.exists(league+'-'+str(rel_year)+'\\'+league+'-'+str(rel_year)+'-May.pckl'):
                    tkMessageBox.showinfo("ATTENTION!!!!", "No data for year "+str(rel_year)+" to get examples")
                    return
        E = EXHandler(league)
        progressbar = ttk.Progressbar(orient=HORIZONTAL, length=200, mode='indeterminate')
        progressbar.pack(side="bottom")
        progressbar.start()
        if len(year.split('-')) == 1:
            thread = myThread(E.get,year)
            thread.start()
            thread.join()
            ex, ta = thread.ex, thread.ta
        else:
            ex, ta = E.get()
        progressbar.stop()
        with open(path.join("GUI_OUTPUT/"+file_name+"_E.pckl"),'w') as res:
            dump(ex,res)
        with open(path.join("GUI_OUTPUT/"+file_name+"_T.pckl"),'w') as res:
            dump(ta,res)
            
def return_active(list,selection):
    if list.get(ACTIVE) == "all":
        now = datetime.datetime.now()
        curr_year = now.year
        selection.set(str(curr_year-5)+"-"+str(curr_year-1))
    elif list.get(ACTIVE) == "current":
        now = datetime.datetime.now()
        curr_year = now.year
        selection.set(str(curr_year))
    else:
        selection.set(list.get(ACTIVE))
    
class SampleApp(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        Tk.geometry(self, "600x400")
        
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (home_page, crawler_page, examples_page):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(home_page)

    def show_frame(self, c):
        '''Show a frame for the given class'''
        frame = self.frames[c]
        frame.tkraise()


class home_page(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        background_image = PhotoImage(file = os.path.join("GUI_Data","GUI_BG.gif"))
        BGLabel = Label(self, image = background_image)
        BGLabel.image = background_image
        BGLabel.place(x=0, y=0, relwidth=1, relheight=1)
        
        label = Label(self, text="This is the AI Project GUI", font=TITLE_FONT)
        label.pack(side="top", fill="x", pady=10)
        
        crawler_b = Button(self, text="Crawler", command=lambda: controller.show_frame(crawler_page),font=BUTTON_FONT)
        examples_b = Button(self, text="Examples Handler", command=lambda: controller.show_frame(examples_page),font=BUTTON_FONT)
        quit_b = Button(self,text="QUIT",bg="red",command=self.quit,font=BUTTON_FONT)
        crawler_b.place(x=170,y=100)
        examples_b.place(x=310,y=100)
        quit_b.pack(side=BOTTOM,pady=30)

class examples_page(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        background_image = PhotoImage(file = os.path.join("GUI_Data","GUI_BG.gif"))
        BGLabel = Label(self, image = background_image)
        BGLabel.image = background_image
        BGLabel.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.label = Label(self, text="This is The Examples Handler", font=TITLE_FONT)
        self.label.pack(side="top", fill="x", pady=10)
        self.label_league = Label(self,height=5,width=38)
        self.label_league.place(x=13,y=90)
        self.label_year = Label(self,height=5,width=38)
        self.label_year.place(x=292,y=90)
        self.label_league_selection = Label(self,height=1,width=38)
        self.label_league_selection.place(x=13,y=210)
        self.label_year_selection = Label(self,height=1,width=38)
        self.label_year_selection.place(x=292,y=210)
        
        self.home_b = Button(self, text="Go to the home page",command=lambda: controller.show_frame(home_page),font=BUTTON_FONT)
        self.home_b.pack(side=BOTTOM,pady=30)
        
        self.select_league_l = Label(self,text = "Please select a"+'\n'+" League :  ",font=SELECTION_FONT)
        self.select_league_l.place(x=13,y=90)
        self.league_list = Listbox(self,height=4)
        with open(os.path.join("GUI_Data","GUI_Leagues.pckl"),'r') as res:
            all_leagues = load(res)
        for league in all_leagues:
            self.league_list.insert(END,league)
        self.league_list.place(x=140,y=92)
        self.sb1 = Scrollbar(self,orient=VERTICAL)
        self.sb1.place(x=265,y=94)
        self.sb1.configure(command=self.league_list.yview)
        self.league_list.configure(yscrollcommand=self.sb1.set)
        
        self.select_year_l = Label(self,text = "Please select a"+'\n'+" Year :  ",font=SELECTION_FONT)
        self.select_year_l.place(x=292,y=90)
        self.year_list = Listbox(self,height=4)
        now = datetime.datetime.now()
        curr_year = now.year
        all_years = range(curr_year-5,curr_year)
        all_years.reverse()
        for year in all_years:
            self.year_list.insert(END,year)
        self.year_list.insert(0,"all")
        self.year_list.insert(1,"current")
        self.year_list.place(x=407,y=92)
        self.sb2 = Scrollbar(self,orient=VERTICAL)
        self.sb2.place(x=532,y=94)
        self.sb2.configure(command=self.year_list.yview)
        self.year_list.configure(yscrollcommand=self.sb2.set)
        
              
        self.select_league_b = Button(self, text="SELECT LEAGUE",bg="cyan",command=lambda: return_active(self.league_list,self.league_selection),font=BUTTON_FONT)
        self.select_league_b.place(x=77,y=175)
        self.selected_league_l = Label(self,text="Selected League is :",font=SELECTION_FONT)
        self.selected_league_l.place(x=13,y=210)
        self.league_selection = StringVar()
        self.league_selection.set("Nothing Selected Yet")
        self.league_entry = Entry(self,textvariable=self.league_selection)
        self.league_entry.place(x=152,y=210)
        
        self.select_year_b = Button(self, text="SELECT YEAR",bg="cyan",command=lambda: return_active(self.year_list,self.year_selection),font=BUTTON_FONT)
        self.select_year_b.place(x=365,y=175)
        self.selected_year_l = Label(self,text="Selected Year is :",font=SELECTION_FONT)
        self.selected_year_l.place(x=292,y=210)
        self.year_selection = StringVar()
        self.year_selection.set("Nothing Selected Yet")
        self.year_entry = Entry(self,textvariable=self.year_selection)
        self.year_entry.place(x=415,y=210)
        
        self.selected_file_l = Label(self,text="Please Enter file name to save :",font=SELECTION_FONT)
        self.selected_file_l.place(x=13,y=260)
        self.selected_file_selection = StringVar()
        self.selected_file_selection.set("")
        self.selected_file_entry = Entry(self,textvariable=self.selected_file_selection)
        self.selected_file_entry.place(x=230,y=260)
        
        self.start_crawl = Button(self, text="GET EXAMPLES & TAGS",bg="green",command=lambda : get_examples(self.league_selection.get(), self.year_selection.get(),self.selected_file_entry.get()),font=BUTTON_FONT)
        self.start_crawl.place(x=370,y=260)
        
        
class crawler_page(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        background_image = PhotoImage(file = os.path.join("GUI_Data","GUI_BG.gif"))
        BGLabel = Label(self, image = background_image)
        BGLabel.image = background_image
        BGLabel.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.label = Label(self, text="This is The Crawler", font=TITLE_FONT)
        self.label.pack(side="top", fill="x", pady=10)
        self.label_league = Label(self,height=5,width=38)
        self.label_league.place(x=13,y=90)
        self.label_year = Label(self,height=5,width=38)
        self.label_year.place(x=292,y=90)
        self.label_league_selection = Label(self,height=1,width=38)
        self.label_league_selection.place(x=13,y=210)
        self.label_year_selection = Label(self,height=1,width=38)
        self.label_year_selection.place(x=292,y=210)
        
        self.home_b = Button(self, text="Go to the home page",command=lambda: controller.show_frame(home_page),font=BUTTON_FONT)
        self.home_b.pack(side=BOTTOM,pady=30)
        
        self.select_league_l = Label(self,text = "Please select a"+'\n'+" League :  ",font=SELECTION_FONT)
        self.select_league_l.place(x=13,y=90)
        self.league_list = Listbox(self,height=4)
        with open(os.path.join("GUI_Data","GUI_Leagues.pckl"),'r') as res:
            all_leagues = load(res)
        for league in all_leagues:
            self.league_list.insert(END,league)
        self.league_list.place(x=140,y=92)
        self.sb1 = Scrollbar(self,orient=VERTICAL)
        self.sb1.place(x=265,y=94)
        self.sb1.configure(command=self.league_list.yview)
        self.league_list.configure(yscrollcommand=self.sb1.set)
        
        self.select_year_l = Label(self,text = "Please select a"+'\n'+" Year :  ",font=SELECTION_FONT)
        self.select_year_l.place(x=292,y=90)
        self.year_list = Listbox(self,height=4)
        now = datetime.datetime.now()
        curr_year = now.year
        all_years = range(curr_year-5,curr_year)
        all_years.reverse()
        for year in all_years:
            self.year_list.insert(END,year)
        self.year_list.insert(0,"all")
        self.year_list.insert(1,"current")
        self.year_list.place(x=407,y=92)
        self.sb2 = Scrollbar(self,orient=VERTICAL)
        self.sb2.place(x=532,y=94)
        self.sb2.configure(command=self.year_list.yview)
        self.year_list.configure(yscrollcommand=self.sb2.set)
        
              
        self.select_league_b = Button(self, text="SELECT LEAGUE",bg="cyan",command=lambda: return_active(self.league_list,self.league_selection),font=BUTTON_FONT)
        self.select_league_b.place(x=77,y=175)
        self.selected_league_l = Label(self,text="Selected League is :",font=SELECTION_FONT)
        self.selected_league_l.place(x=13,y=210)
        self.league_selection = StringVar()
        self.league_selection.set("Nothing Selected Yet")
        self.league_entry = Entry(self,textvariable=self.league_selection)
        self.league_entry.place(x=152,y=210)
        
        self.select_year_b = Button(self, text="SELECT YEAR",bg="cyan",command=lambda: return_active(self.year_list,self.year_selection),font=BUTTON_FONT)
        self.select_year_b.place(x=365,y=175)
        self.selected_year_l = Label(self,text="Selected Year is :",font=SELECTION_FONT)
        self.selected_year_l.place(x=292,y=210)
        self.year_selection = StringVar()
        self.year_selection.set("Nothing Selected Yet")
        self.year_entry = Entry(self,textvariable=self.year_selection)
        self.year_entry.place(x=415,y=210)
        
        self.leagues_links = {'Primer_League':"http://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League",
                     'Serie_A':"http://www.whoscored.com/Regions/108/Tournaments/5/Italy-Serie-A",
                     'La_Liga':"http://www.whoscored.com/Regions/206/Tournaments/4/Spain-La-Liga",
                     'Bundesliga':"http://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga",
                     'Ligue_1':"http://www.whoscored.com/Regions/74/Tournaments/22/France-Ligue-1"
                     }
        
        args = ["python","whoscored_crawler.py"]
        self.start_crawl = Button(self, text="START CRAWLING",bg="green",command=lambda : start_crawl_func(args, self.league_selection.get(), self.year_selection.get()),font=BUTTON_FONT)
        self.start_crawl.pack(side=BOTTOM,pady=31)
        
        

    
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()