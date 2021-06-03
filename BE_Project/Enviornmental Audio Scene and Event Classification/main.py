# Import files
from definations import *
from save_stft import *
from feature_extraction import *
from get_data import *
from build_model import *
from noise_reduction import *
from split_audio import *
from prediction import *
from exit import *
from external_links import *

#FIRST WINDOW STARTS FROM HERE

main_ui.configure(bg='#303030')

main_ui.title("Environmental Audio Scene and Sound Event Recogntion")
main_ui.geometry('1366x768')

Wel_Label = Label(main_ui,text='Welcome !\n', fg="white",bg="#303030", font=('Arial', 20)).pack()

MainLabel = Label(main_ui,text='Select the process\n\n1.Extract Features from Audio samples\n\n2.Build Model\n\n3.Check a sound file\n\n4.Visualize an audio file processing before feature extraction\n\n5.Exit\n', fg="white",bg="#303030", font=('opensans', 12,'bold')).pack()

feature_extr_button = Button(main_ui,text="Feature Extraction", fg="black", bg="#FFEF00", command=Feature_Extraction, font=('opensans', 12, 'bold'),border=0,width=20).place(x=50, y=325)

model_build_button = Button(main_ui,text="Build Model", fg="black", bg="#FFEF00", command=Build_and_save_model, font=('opensans', 12, 'bold'),border=0,width=20).place(x=590, y=325)

predict_button = Button(main_ui,text="Prediction", fg="black", bg="#FFEF00", command=run_prediction, font=('opensans', 12, 'bold'),border=0,width=20).place(x=1120, y=325)

visualizer_button = Button(main_ui,text="Visualize", fg="black", bg="#FFEF00", command=noise_reduction, font=('opensans', 12, 'bold'),border=0,width=20).place(x=310, y=400)

exit_button = Button(main_ui,text='Exit', fg="black", bg='#FFEF00',command=exit, font=('opensans', 12, 'bold'),border=0,width=20).place(x=860, y=400)

# External Links

Label(main_ui,text='\n\nMore on effects of noise on children\n', fg="white",bg="#303030", font=('opensans', 13)).place(x=550,y=470)

link1_button = Button(main_ui,text='Noise & Its effects on children', fg="#7DF9FF", bg='#303030',command=link1, font=('opensans', 12),border=0,width=30,activebackground="#303030").place(x=20, y=600)

link2_button = Button(main_ui,text='Children and Noise', fg="#7DF9FF", bg='#303030',command=link2, font=('opensans', 12),border=0,width=30,activebackground="#303030").place(x=360, y=600)

link3_button = Button(main_ui,text='infuences of background noise', fg="#7DF9FF", bg='#303030',command=link3, font=('opensans', 12),border=0,width=30,activebackground="#303030").place(x=710, y=600)

link4_button = Button(main_ui,text='Noise as a health hazard', fg="#7DF9FF", bg='#303030',command=link4, font=('opensans', 12),border=0,width=30,activebackground="#303030").place(x=1050, y=600)

main_ui.mainloop()