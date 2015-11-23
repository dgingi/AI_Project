from Tkinter import *
from crawler import *
from pickle import load
import subprocess
import tkMessageBox
import Tkinter
  
TITLE_FONT = ("Helvetica", 18, "bold")

def start_crawl_func(args,league,year):
    if league == "Nothing Selected Yet" or year == "Nothing Selected Yet":
        tkMessageBox.showinfo("ATTENTION!!!!", "Please select both league and year")
    else:
        subprocess.Popen(args+[league,year])
    
class SampleApp(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        Tk.geometry(self, "600x400")
        
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (home_page, crawler_page):
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
        
        background_image = PhotoImage(file = "C:\Users\dror\Documents\Technion\AI-Project\GUI_BG.gif")
        BGLabel = Label(self, image = background_image)
        BGLabel.image = background_image
        BGLabel.place(x=0, y=0, relwidth=1, relheight=1)
        
        label = Label(self, text="This is the AI Project GUI", font=TITLE_FONT)
        label.pack(side="top", fill="x", pady=10)
        
        crawler_b = Button(self, text="Crawler",
                            command=lambda: controller.show_frame(crawler_page))
        quit_b = Button(self,text="QUIT",bg="red",command=self.quit)
        crawler_b.pack()
        quit_b.pack(side=BOTTOM)


class crawler_page(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        background_image = PhotoImage(file = "C:\Users\dror\Documents\Technion\AI-Project\GUI_BG.gif")
        BGLabel = Label(self, image = background_image)
        BGLabel.image = background_image
        BGLabel.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.label = Label(self, text="This is The crawler", font=TITLE_FONT)
        self.label.pack(side="top", fill="x", pady=10)
        self.label_league = Label(self,height=5,width=38)
        self.label_league.place(x=13,y=90)
        self.label_year = Label(self,height=5,width=38)
        self.label_year.place(x=292,y=90)
        self.label_league_selection = Label(self,height=1,width=35)
        self.label_league_selection.place(x=13,y=210)
        self.label_year_selection = Label(self,height=1,width=33)
        self.label_year_selection.place(x=292,y=210)
        
        self.home_b = Button(self, text="Go to the home page",
                           command=lambda: controller.show_frame(home_page))
        self.home_b.pack(side=BOTTOM,pady=30)
        
        self.select_league_l = Label(self,text = "Please select a League :  ")
        self.select_league_l.place(x=13,y=90)
        self.league_list = Listbox(self,height=4)
        all_leagues = ["Primer_League","Seria_A","La_Liga","Bundesliga","Ligue_1"]
        for league in all_leagues:
            self.league_list.insert(END,league)
        self.league_list.place(x=140,y=92)
        self.sb1 = Scrollbar(self,orient=VERTICAL)
        self.sb1.place(x=265,y=94)
        self.sb1.configure(command=self.league_list.yview)
        self.league_list.configure(yscrollcommand=self.sb1.set)
        
        self.select_year_l = Label(self,text = "Please select a Year :  ")
        self.select_year_l.place(x=292,y=90)
        self.year_list = Listbox(self,height=4)
        all_years = range(2010,2015)
        all_years.reverse()
        for year in all_years:
            self.year_list.insert(END,year)
        self.year_list.insert(END,"all")
        self.year_list.place(x=407,y=92)
        self.sb2 = Scrollbar(self,orient=VERTICAL)
        self.sb2.place(x=532,y=94)
        self.sb2.configure(command=self.year_list.yview)
        self.year_list.configure(yscrollcommand=self.sb2.set)
        
              
        self.select_league_b = Button(self, text="SELECT LEAGUE",bg="cyan",command=lambda: self.return_active(self.league_list,self.league_selection))
        self.select_league_b.place(x=140,y=175)
        self.selected_league_l = Label(self,text="Selected League is :")
        self.selected_league_l.place(x=13,y=210)
        self.league_selection = StringVar()
        self.league_selection.set("Nothing Selected Yet")
        self.league_entry = Entry(self,textvariable=self.league_selection)
        self.league_entry.place(x=140,y=210)
        
        self.select_year_b = Button(self, text="SELECT YEAR",bg="cyan",command=lambda: self.return_active(self.year_list,self.year_selection))
        self.select_year_b.place(x=407,y=175)
        self.selected_year_l = Label(self,text="Selected Year is :")
        self.selected_year_l.place(x=292,y=210)
        self.year_selection = StringVar()
        self.year_selection.set("Nothing Selected Yet")
        self.year_entry = Entry(self,textvariable=self.year_selection)
        self.year_entry.place(x=407,y=210)
        
        self.leagues_links = {'Primer_League':"http://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League",
                     'Serie_A':"http://www.whoscored.com/Regions/108/Tournaments/5/Italy-Serie-A",
                     'La_Liga':"http://www.whoscored.com/Regions/206/Tournaments/4/Spain-La-Liga",
                     'Bundesliga':"http://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga",
                     'Ligue_1':"http://www.whoscored.com/Regions/74/Tournaments/22/France-Ligue-1"
                     }
        
        args = ["python","crawler.py"]
        self.start_crawl = Button(self, text="START CRAWLING",bg="green",command=lambda : start_crawl_func(args, self.league_selection.get(), self.year_selection.get()))
        self.start_crawl.pack(side=BOTTOM,pady=40)
        
        

    def return_active(self,list,selection):
        if list.get(ACTIVE) == "all":
            selection.set("2010-2014")
        else:
            selection.set(list.get(ACTIVE))
    
    
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()