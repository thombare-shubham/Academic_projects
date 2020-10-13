import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tkinter import *

main_ui = Tk()

user_reviews = pd.read_csv("C:\MY FILES\Project\googleplaystore_user_reviews.csv")
playstore_data = pd.read_csv("C:\MY FILES\Project\googleplaystore.csv")
store_data = pd.read_csv("C:\MY FILES\Project\AppleStore.csv")
app_desc = pd.read_csv("C:\MY FILES\Project\AppleStore_description.csv")

# Definations of All programs
 # removing the missing values
print("Performing data Cleaning!\n")
user_reviews.dropna(inplace=True)
# remove duplicates in play store1

# playstore_data.drop_duplicates(inplace=True)
playstore_data = playstore_data.drop_duplicates(subset='App')
playstore_data.dropna(inplace=True, subset=['Type', 'Content Rating', 'Current Ver', 'Android Ver'])
# filling rating in missing areas by not captured (0)

playstore_data.fillna(0, inplace=True)
playstore_data[playstore_data['Rating'] == 0].head()



def googleplaystore():
    play_ui = Tk()
    play_ui.configure(bg='#ef5777')
    # pl_data_cleaning()
    # Second Slide Data
    play_ui.title("Appylyzer:Google Playstore Ananlysis!")
    play_ui.geometry("850x550+50+50")
    ws_label = Label(play_ui, text='Google Play Store!', fg="black", bg="#ef5777", font=('Arial', 20, 'bold')).pack()
    wsbutton_one = Button(play_ui, text="1.Find total no of apps & reviews", fg="black", bg="#f53b57",
                          command=pl_total, font=('opensans', 12, 'italic')).place(x=170, y=50)
    wsbutton_two = Button(play_ui, text="2.Descriptive Analysis on Play store", fg="black", bg="#f53b57",
                          command=pl_desc_analysis, font=('opensans', 12, 'italic')).place(x=170, y=100)
    wsbutton_three = Button(play_ui, text="3.User Review Bar plot Categorical view", fg="black", bg="#f53b57",
                            command=pl_user_review_bar_plot, font=('opensans', 12, 'italic')).place(x=170, y=150)
    wsbutton_four = Button(play_ui, text="4.Bar chart on Overall Play store", fg="black", bg="#f53b57",
                           command=pl_overall_analysis, font=('opensans', 12, 'italic')).place(x=170, y=200)
    wsbutton_five = Button(play_ui, text="5.Bar chart On Genres", fg="black", bg="#f53b57",
                           command=pl_generic_analysis, font=('opensans', 12, 'italic')).place(x=170, y=250)
    wsbutton_six = Button(play_ui, text="6.View Apps with 1 billion downloads", fg="black", bg="#f53b57",
                          command=pl_billion_exceed, font=('opensans', 12, 'italic')).place(x=170, y=300)
    wsbutton_seven = Button(play_ui, text="7.View top 10 Apps In Each Segment", fg="black", bg="#f53b57",
                            command=pl_each_seg_top, font=('opensans', 12, 'italic')).place(x=170, y=350)
    wsbutton_eight = Button(play_ui, text="8.analysis on paid apps", fg="black", bg="#f53b57",
                            command=pl_paid_analysis, font=('opensans', 12, 'italic')).place(x=170, y=400)
    wsbutton_nine = Button(play_ui, text="9.Analyze Free vs Paid Apps", fg="black", bg="#f53b57",
                           command=pl_freevspaid, font=('opensans', 12, 'italic')).place(x=170, y=450)
    wsbutton_ten = Button(play_ui, text="10.Exit", fg="black", bg="#f53b57", command=play_ui.destroy,
                          font=('opensans', 12, 'italic')).place(x=170, y=500)
    play_ui.mainloop()
    #exit()


def applestore():
    apple_ui = Tk()
    apple_ui.configure(bg='#d2dae2')
    apple_ui.title("Appylyzer:Apple App store Ananlysis!")
    apple_ui.geometry("850x550+50+50")
    wt_label = Label(apple_ui, text='Apple App Store!', fg="black", bg="#d2dae2", font=('arial', 20, 'bold')).pack()
    wtbutton_one = Button(apple_ui, text="1.View Head Entries and description about Dataset", fg="black", bg="#95a5a6",
                          command=as_apple_head, font=('opensans', 12, 'italic')).place(x=170, y=50)
    wtbutton_two = Button(apple_ui, text="2.Top 10 Apps based on Rating", fg="black", bg="#95a5a6",
                          command=as_top_on_rating, font=('opensans', 12, 'italic')).place(x=170, y=100)
    wtbutton_three = Button(apple_ui, text="3.Top 10 Apps Based on download Size", fg="black", bg="#95a5a6",
                            command=as_top_on_download, font=('opensans', 12, 'italic')).place(x=170, y=150)
    wtbutton_four = Button(apple_ui, text="4.Top 10 apps on basis of price", fg="black", bg="#95a5a6",
                           command=as_top_on_price, font=('opensans', 12, 'italic')).place(x=170, y=200)
    wtbutton_five = Button(apple_ui, text="5.User favourites of all time", fg="black", bg="#95a5a6",
                           command=as_all_time_fav, font=('opensans', 12, 'italic')).place(x=170, y=250)
    wtbutton_six = Button(apple_ui, text="6.User favourites of current time", fg="black", bg="#95a5a6",
                          command=as_current_fav, font=('opensans', 12, 'italic')).place(x=170, y=300)
    wtbutton_seven = Button(apple_ui, text="7.Overall User rating review", fg="black", bg="#95a5a6",
                            command=as_rating_analysis, font=('opensans', 12, 'italic')).place(x=170, y=350)
    wtbutton_eight = Button(apple_ui, text="8.Exit", fg='black', bg="#95a5a6", command=apple_ui.destroy,
                            font=('opensans', 12, 'italic')).place(x=170, y=400)
    apple_ui.mainloop()
    #exit()


# Function For Visualization
def visualizer(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15, 8)):
    plt.figure(figsize=figsize)

    if plot_type == "bar":
        sns.barplot(x=x, y=y)
    elif plot_type == "count":
        sns.countplot(x)
    elif plot_type == "reg":
        sns.regplot(x=x, y=y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13, rotation=rotation_value)
    plt.show()


def pl_total():
    print("PlayStore data size is ", playstore_data.shape)
    print("User reviews data size is ", user_reviews.shape)


def pl_desc_analysis():
    # descriptive analysis on play store

    print(playstore_data.describe())
    # density plot shape
    rating = playstore_data[playstore_data['Rating'] != 0]
    print("After removing the missing values in ratings")
    print(rating.describe())
    # sns.kdeplot(shade=True, data=rating['Rating'])

    from scipy.stats import kurtosis, skew

    x = np.random.normal(0, 2, 10000)
    print('excess kurtosis of  distribution : {}'.format(kurtosis(rating['Rating'])))
    print('skewness of distribution: {}'.format(skew(rating['Rating'])))


def pl_user_review_bar_plot():
    # Bar plot on categorical variable

    df1 = user_reviews['Sentiment'].value_counts()
    df1 = df1.reset_index()

    def bar_plot(x, y, y_label, title, color):
        objects = x.values
        y_pos = np.arange(len(objects))
        plt.figure(figsize=(10, 5))
        bar = plt.bar(x, y, color=color)
        plt.xticks(y_pos, objects)
        plt.ylabel(y_label)
        plt.title(title)

        return bar

    print(df1['index'].values)

    bar_plot(x=df1['index'], y=df1['Sentiment'], color='g', y_label='Sentiment_Freq', title='Bar Plot on Sentiment')


def pl_overall_analysis():
    list_1 = ['Category', 'Installs', 'Type',
              'Content Rating']

    def bar_plot(x, y, y_label, x_label, title, color, ax):
        # plt.figure(figsize=(10,5))
        bar = sns.barplot(x=x, y=y, ax=ax, orient='h')
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        for i, v in enumerate(x.values):
            ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
        return bar

    fig = plt.figure(figsize=(14, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    i = 1
    for names in list_1:
        ax1 = fig.add_subplot(2, 2, i)
        df2 = playstore_data[names].value_counts()
        df2 = df2.reset_index()
        bar_plot(x=df2[names], y=df2['index'], y_label='Freq', title='Bar Chart On {}'.format(names), color='red',
                 ax=ax1, x_label=names)
        i += 1


def pl_generic_analysis():
    list_2 = ['Genres']

    def bar_plot(x, y, y_label, x_label, title, color, ax=None):
        plt.figure(figsize=(5, 8))
        bar = sns.barplot(x=x, y=y, orient='h')
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        for i, v in enumerate(x.values):
            bar.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
        return bar

    df2 = playstore_data['Genres'].value_counts()
    df2 = df2.reset_index()
    df2 = df2[df2['Genres'] > 100]
    bar_plot(x=df2['Genres'], y=df2['index'], y_label='Freq', title='Bar Chart On Gerner', color='red',
             x_label='Genre')


def pl_billion_exceed():
    print("Apps with 1 Billion Downloads:\n")
    print(playstore_data[playstore_data['Installs'] == '1,000,000,000+']['App'])


def pl_each_seg_top():
    df2 = playstore_data['Genres'].value_counts()
    df2 = df2.reset_index()
    df2 = df2[df2['Genres'] > 100]
    genres = list(df2['index'][1:10])
    d = pd.DatetimeIndex(playstore_data['Last Updated'])
    playstore_data['year'] = d.year
    playstore_data['month'] = d.month
    for i in genres:
        play = playstore_data[(playstore_data['Installs'] != '1,000,000,000+') & (playstore_data['Genres'] == i) & (
                playstore_data['Rating'] >= 4.5) & (playstore_data['year'] == 2018)]['App']
        print('')
        print('Printing 10 Apps with 100 million installs and Rating >= 4.5 and Year = 2018 in {}'.format(i))
        print('--------------------------------------------------')
        print(play[0:10])


def pl_paid_analysis():
    # analysis on paid apps
    print("Top paid apps are:")
    paided = playstore_data[playstore_data['Type'] == 'Paid']
    df3 = paided['Category'].value_counts()
    df3 = df3.reset_index()
    df3 = df3[:10]
    plt.figure(figsize=(10, 5))
    plt.pie(x=list(df3['Category']), labels=list(df3['index']), autopct='%1.0f%%', pctdistance=0.8,
            labeldistance=1.2)
    plt.title('% Distribution of Paided Apps Categories')

    # Top rated paid apps with installs 1,000,000+

    print(paided[(paided['Rating'] > 4.7) & (paided['Installs'] == '100,000+')]['App'])


def pl_freevspaid():
    # % free vs paid apps

    size = [8895, 753]
    sentiment = ['Free', 'Paid']
    colors = ['g', 'pink']
    plt.pie(size, labels=sentiment, colors=colors, startangle=180, autopct='%.1f%%')
    plt.title('% Free vs Paid Apps')
    plt.show()


def exit():
    end_ui = Tk()
    end_ui.configure(bg='#d1d8e0')
    end_ui.title("Thanking you!")
    end_ui.geometry('768x400+100+100')
    end_label = Label(end_ui,
                      text='Presented By:\n1.Ankush Soni\nKunal Sonar\n3.Shubham Thombare\n\n\nThank You!',
                      bg='#d1d8e0', fg='black', font=('opensans', 20, 'italic')).pack()
    # print("Escaped!\n")
    #sys.exit
    main_ui.destroy()
    MainButton2 = Button(end_ui,text="Exit", fg="black", bg="#778ca3", command=end_ui.destroy,
                         font=('arial', 12, 'italic')).place(x=330, y=350)


    end_ui.mainloop()

# Apple store Functions
def as_apple_head():
    print("First 5 entries in Dataset Applestore\n")
    print(store_data.head())
    print("Info About Applestore Dataset\n")
    store_data.info()
    print("First 5 entries in Dataset Apple Description\n")
    print(app_desc.head())
    print("Info About Applestore_description Dataset\n")
    app_desc.info()


def as_top_on_rating():
    store_data_sorted = store_data.sort_values('rating_count_tot', ascending=False)
    subset_store_data_sorted = store_data_sorted[:10]

    visualizer(subset_store_data_sorted.track_name, subset_store_data_sorted.rating_count_tot, "bar",
               "TOP 10 APPS ON THE BASIS OF TOTAL RATINGS",
               "APP NAME", "RATING COUNT (TOTAL)", True, -60)


def as_top_on_download():
    store_data_download = store_data.sort_values('size_bytes', ascending=False)
    store_data_download.size_bytes /= 1024 * 1024  # Conversion from Bytes to MegaBytes
    subset_store_data_download = store_data_download[:10]

    visualizer(subset_store_data_download.track_name, subset_store_data_download.size_bytes, "bar",
               "TOP 10 APPS ON THE BASIS OF DOWNLOAD SIZE",
               "APP NAME", "DOWNLOAD SIZE (in MB)", True, -60)


def as_top_on_price():
    store_data_price = store_data.sort_values('price', ascending=False)
    subset_store_data_price = store_data_price[:10]

    visualizer(subset_store_data_price.price, subset_store_data_price.track_name, "bar",
               "TOP 10 APPS ON THE BASIS OF PRICE",
               "Price (in USD)", "APP NAME")


def as_all_time_fav():
    store_data["favourites_tot"] = store_data["rating_count_tot"] * store_data["user_rating"]
    store_data["favourites_ver"] = store_data["rating_count_ver"] * store_data["user_rating_ver"]
    favourite_app = store_data.sort_values("favourites_tot", ascending=False)
    favourite_app_subset = favourite_app[:10]

    visualizer(favourite_app_subset.track_name, favourite_app_subset.rating_count_tot, "bar",
               "FAVOURITES (ALL TIME)",
               "APP NAME", "RATING COUNT(TOTAL)", True, -60)


def as_current_fav():
    store_data["favourites_tot"] = store_data["rating_count_tot"] * store_data["user_rating"]
    store_data["favourites_ver"] = store_data["rating_count_ver"] * store_data["user_rating_ver"]
    favourite_app = store_data.sort_values("favourites_tot", ascending=False)
    favourite_app_subset = favourite_app[:10]
    favourite_app_ver = store_data.sort_values("favourites_ver", ascending=False)
    favourite_app_ver_subset = favourite_app_ver[:10]

    visualizer(favourite_app_ver_subset.rating_count_ver, favourite_app_ver_subset.track_name,
               "bar", "FAVOURITES (CURRENT VERSION)", "RATING COUNT(CURRENT VERSION)", "APP NAME", False)


def as_rating_analysis():
    visualizer(store_data.user_rating, None, "count", "RATINGS ON APP STORE",
               "RAITNGS", "NUMBER OF APPS RATED")


main_ui.title("Appylyzer")
main_ui.geometry("850x550+50+50")
main_ui.configure(bg='#ffaf40')

# First Slide data
Wel_Label = Label(text='Welcome Geek!\n', fg="black", bg="#ffaf40", font=('Arial', 20, 'bold')).pack()
MainLabel = Label(text='Select App Store', fg="black", bg="#ffaf40", font=('Arial', 20, 'bold')).pack()
MainButton1 = Button(text="Google Play Store", fg="black", bg="#fa8231", command=googleplaystore,
                     font=('opensans', 12, 'italic')).place(x=170, y=150)
MainButton2 = Button(text="Apple App Store", fg="black", bg="#fa8231", command=applestore,
                     font=('opensans', 12, 'italic')).place(x=370, y=150)
MainButton3 = Button(text='3.Exit', fg="black", bg="#fa8231", command=exit,
                     font=('opensans', 12, 'italic')).place(x=570, y=150)
#exit()
main_ui.mainloop()
