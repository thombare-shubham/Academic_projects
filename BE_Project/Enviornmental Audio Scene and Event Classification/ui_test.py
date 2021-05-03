from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

from numpy.core.fromnumeric import size

def exit():
    end_ui = Tk()
    end_ui.configure(bg='pink')
    end_ui.title("Thanking you!")
    end_ui.geometry('768x400')
    end_label = Label(end_ui,text='Presented By:\n\n1.Shubham Thombare\n\n2.Vishal Kajale\n\n3.Ankush Soni\n\n4.Kunal Sonar\n\n\nThank You!',bg='pink', fg='black', font=('opensans', 15, 'bold')).pack()
    main_ui.destroy()
    MainButton2 = Button(end_ui, text="Exit", fg="pink", bg="purple", command=end_ui.destroy,border=0,width=15,
                         font=('arial', 12, 'italic')).place(x=330, y=350)

    end_ui.mainloop()

# Flags
feature_flag = 0
model_flag = 0

#FIRST WINDOW STARTS FROM HERE
main_ui = Tk()

main_ui.configure(bg='pink')

main_ui.title("Environmental Audio Scene and Sound Event Recogntion")
main_ui.geometry('850x550')

Wel_Label = Label(text='Welcome Geek!\n', fg="black",bg="pink", font=('Arial', 20, 'bold')).pack()

MainLabel = Label(text='Select the process\n\n1.Extract Features from Audio samples\n\n2.Build Model\n\n3.Check a sound file\n\n4.Visualize an audio file processing before feature extraction\n\n5.Exit\n', fg="black",bg="pink", font=('Arial', 12, 'bold')).pack()

if(feature_flag != 1):
    file_add_button = Button(text="Feature Extraction", fg="pink", bg="purple", command='#', font=('opensans', 12, 'bold'),border=0,width=15).place(x=25, y=325)
    feature_flag = 1
else:
    messagebox.showwarning("Warning","Features Already extracted!")
if(model_flag != 1):
    model_build_button = Button(text="Build Model", fg="pink", bg="purple", command='#', font=('opensans', 12, 'bold'),border=0,width=15).place(x=350, y=325)
    model_flag = 1
else:
    messagebox.showwarning("Warning","Model is already builded!")

prediction_button = Button(text="Prediction", fg="pink", bg="purple", command='#', font=('opensans', 12, 'bold'),border=0,width=15).place(x=675, y=325)

visualizer_button = Button(text="Visualize", fg="pink", bg="purple", command='#', font=('opensans', 12, 'bold'),border=0,width=15).place(x=185, y=400)

exit_button = Button(text='Exit', fg="pink", bg='purple',command=exit, font=('opensans', 12, 'bold'),border=0,width=15).place(x=515, y=400)

main_ui.mainloop()