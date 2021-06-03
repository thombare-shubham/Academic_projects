from definations import *

# EXIT FUNCTION FOR SECOND WINDOW
def exit():
    main_ui.destroy()
    end_ui = Tk()
    end_ui.configure(bg='#303030')
    end_ui.title("Thanking you!")
    end_ui.geometry('1366x768')
    end_label = Label(end_ui,text='Presented By:\n\n1.Shubham Thombare\n\n2.Vishal Kajale\n\n3.Ankush Soni\n\n4.Kunal Sonar\n\nUnder Guidance of "Prof. Pradenesh Bhisikar"\n\n\nThank You!',bg='#303030', fg='white', font=('opensans', 12,'bold')).pack()
    MainButton2 = Button(end_ui, text="Exit", fg="black", bg="#FFEF00", command=end_ui.destroy,border=0,width=15,font=('arial', 12,'bold')).pack(padx=20,pady=30)

    end_ui.mainloop()