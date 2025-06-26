from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageFilter, Image
import os
import psutil
import time
import subprocess
import cv2
import fnmatch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Global suggestion lists for each meal category.
breakfast_suggestions = []
lunch_suggestions = []
dinner_suggestions = []

main_win = Tk()
# main_win.iconbitmap('3/favicon.ico')

def show_entry_fields():
    print(" Age: %s\n Veg-NonVeg: %s\n Weight: %s\n Height: %s\n" % 
          (e1.get(), e2.get(), e3.get(), e4.get()))

################################################################################
# Weight Loss Function
################################################################################
def Weight_Loss():
    print(" Age: %s\n Veg-NonVeg: %s\n Weight: %s\n Height: %s\n" % 
          (e1.get(), e2.get(), e3.get(), e4.get()))
    print("\n===== DIET RECOMMENDATION: Weight Loss =====")
    print("User Input:")
    print(f"  Age       : {e1.get()}")
    print(f"  Veg/NonVeg: {e2.get()}")
    print(f"  Weight    : {e3.get()}")
    print(f"  Height    : {e4.get()}")

    # Read the CSV file.
    data = pd.read_csv('input.csv')
    data.head(5)
    
    # Extract relevant columns.
    Food_itemsdata = data["Food_items"]
    Breakfastdata = data["Breakfast"]
    Lunchdata = data["Lunch"]
    Dinnerdata = data["Dinner"]
    
    # Partition food items by meal.
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []
    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []
    
    for i in range(len(Breakfastdata)):
        if int(Breakfastdata.iloc[i]) == 1:
            breakfastfoodseparated.append(Food_itemsdata.iloc[i])
            breakfastfoodseparatedID.append(i)
        if int(Lunchdata.iloc[i]) == 1:
            Lunchfoodseparated.append(Food_itemsdata.iloc[i])
            LunchfoodseparatedID.append(i)
        if int(Dinnerdata.iloc[i]) == 1:
            Dinnerfoodseparated.append(Food_itemsdata.iloc[i])
            DinnerfoodseparatedID.append(i)
            
    print('BREAKFAST FOOD ITEMS:', breakfastfoodseparated)
    print('LUNCH FOOD ITEMS:', Lunchfoodseparated)
    print('DINNER FOOD ITEMS:', Dinnerfoodseparated)
    print("\n[MEALS]")
    print("  Breakfast:", breakfastfoodseparated)
    print("  Lunch    :", Lunchfoodseparated)
    print("  Dinner   :", Dinnerfoodseparated)
    
    # Retrieve a subset of columns.
    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID].T.iloc[Valapnd].T
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID].T.iloc[Valapnd].T
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID].T.iloc[Valapnd].T

    print(DinnerfoodseparatedIDdata.describe())
    
    # Get user inputs.
    age = int(e1.get())
    veg = float(e2.get())   # 1 means veg, 0 means nonveg.
    weight = float(e3.get())
    height = float(e4.get())
    
    bmi = weight / (height ** 2)
    print(f"\n[INFO] BMI Calculated: {bmi:.2f}")
    
    # BMI flag.
    if bmi < 16:
        clbmi = 4
    elif bmi < 18.5:
        clbmi = 3
    elif bmi < 25:
        clbmi = 2
    elif bmi < 30:
        clbmi = 1
    else:
        clbmi = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp + 20)
        for i in test_list:
            if i == age:
                print("Age is between", str(lp), "and", str(lp+10))
                agecl = round(lp / 20)
    print("Your BMI is:", bmi)
    
    # Convert meal DataFrames to numpy arrays.
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    
    ti = (clbmi + agecl) / 2
    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)
    
    # KMeans clustering for each meal.
    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dnrlbl = kmeans.labels_
    
    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    lnchlbl = kmeans.labels_
    
    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    brklbl = kmeans.labels_
    
    print("\n[CLUSTERING RESULTS]")
    print("  Dinner Clusters:", list(dnrlbl))
    print("  Lunch Clusters :", list(lnchlbl))
    print("  Breakfast Clusters:", list(brklbl))
    
    # Process training CSV.
    datafin = pd.read_csv('inputfin.csv')
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]].T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]].T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]].T

    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, :]
    weightgaincat = weightgaincatDdata[1:, :]
    healthycat = healthycatDdata[1:, :]
    
    weightlossfin = np.zeros((len(weightlosscat) * 5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat) * 5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat) * 5, 9), dtype=np.float32)
    
    t = r = s = 0
    yt, yr, ys = [], [], []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightlosscat), 6), dtype=np.float32)
    for jj in range(len(weightlosscat)):
        valloc = list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc) * ti

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(weightlossfin, yt)
    y_pred = clf.predict(X_test)
    
    print("\n[MODEL PREDICTION]")
    print("  Predicted Labels:", list(y_pred))
    
    # Filter recommendations using VegNovVeg column. For veg users (veg==1),
    # only include food items where VegNovVeg==0.
    global breakfast_suggestions, lunch_suggestions, dinner_suggestions
    breakfast_suggestions = []
    lunch_suggestions = []
    dinner_suggestions = []
    max_items = min(len(y_pred), len(Food_itemsdata))
    for ii in range(max_items):
        if y_pred[ii] == 2:
            food_item = Food_itemsdata.iloc[ii]
            if int(veg) == 1 and int(data["VegNovVeg"].iloc[ii]) != 0:
                print("Skipping non-veg item:", food_item)
                continue
            # Add exclusively to breakfast if applicable.
            if ii in breakfastfoodseparatedID:
                breakfast_suggestions.append(food_item)
                print("  - Breakfast:", food_item)
            else:
                # Allow duplicates in lunch and dinner.
                if ii in LunchfoodseparatedID:
                    lunch_suggestions.append(food_item)
                    print("  - Lunch:", food_item)
                if ii in DinnerfoodseparatedID:
                    dinner_suggestions.append(food_item)
                    print("  - Dinner:", food_item)
    if not (breakfast_suggestions or lunch_suggestions or dinner_suggestions):
        print("  No suitable food items found.")
    print("===========================================\n")

################################################################################
# Healthy Function
################################################################################
def Healthy():
    print(" Age: %s\n Veg-NonVeg: %s\n Weight: %s\n Height: %s\n" % 
          (e1.get(), e2.get(), e3.get(), e4.get()))
    print("\n===== DIET RECOMMENDATION: Healthy =====")
    print("User Input:")
    print("  Age       :", e1.get())
    print("  Veg/NonVeg:", e2.get())
    print("  Weight    :", e3.get())
    print("  Height    :", e4.get())
    
    data = pd.read_csv('input.csv')
    data.head(5)
    Food_itemsdata = data["Food_items"]
    Breakfastdata = data["Breakfast"]
    Lunchdata = data["Lunch"]
    Dinnerdata = data["Dinner"]
    
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []
    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []
    
    for i in range(len(Breakfastdata)):
        if int(Breakfastdata.iloc[i]) == 1:
            breakfastfoodseparated.append(Food_itemsdata.iloc[i])
            breakfastfoodseparatedID.append(i)
        if int(Lunchdata.iloc[i]) == 1:
            Lunchfoodseparated.append(Food_itemsdata.iloc[i])
            LunchfoodseparatedID.append(i)
        if int(Dinnerdata.iloc[i]) == 1:
            Dinnerfoodseparated.append(Food_itemsdata.iloc[i])
            DinnerfoodseparatedID.append(i)
    
    print("BREAKFAST FOOD ITEMS:", breakfastfoodseparated)
    print("LUNCH FOOD ITEMS:", Lunchfoodseparated)
    print("DINNER FOOD ITEMS:", Dinnerfoodseparated)
    
    print("\n[MEALS]")
    print("  Breakfast:", breakfastfoodseparated)
    print("  Lunch    :", Lunchfoodseparated)
    print("  Dinner   :", Dinnerfoodseparated)
    
    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID].T.iloc[Valapnd].T
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID].T.iloc[Valapnd].T
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID].T.iloc[Valapnd].T
    
    age = int(e1.get())
    veg = float(e2.get())
    weight = float(e3.get())
    height = float(e4.get())
    bmi = weight / (height ** 2)
    print(f"\n[INFO] BMI Calculated: {bmi:.2f}")
    
    if bmi < 16:
        clbmi = 4
    elif bmi < 18.5:
        clbmi = 3
    elif bmi < 25:
        clbmi = 2
    elif bmi < 30:
        clbmi = 1
    else:
        clbmi = 0
    
    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp + 20)
        for i in test_list:
            if i == age:
                print("Age is between", str(lp), "and", str(lp+10))
                agecl = round(lp / 20)
    print("Your BMI is:", bmi)
    
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi + agecl) / 2
    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)
    
    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dnrlbl = kmeans.labels_
    
    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    lnchlbl = kmeans.labels_
    
    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    brklbl = kmeans.labels_
    
    print("\n[CLUSTERING RESULTS]")
    print("  Dinner Clusters:", list(dnrlbl))
    print("  Lunch Clusters :", list(lnchlbl))
    print("  Breakfast Clusters:", list(brklbl))
    
    datafin = pd.read_csv('inputfin.csv')
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1,2,7,8]].T
    weightgaincat = dataTog.iloc[[0,1,2,3,4,7,9,10]].T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]].T

    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()

    weightlosscat = weightlosscatDdata[1:, :]
    weightgaincat = weightgaincatDdata[1:, :]
    healthycat = healthycatDdata[1:, :]

    weightlossfin = np.zeros((len(weightlosscat) * 5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat) * 5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat) * 5, 9), dtype=np.float32)

    t = r = s = 0
    yt, yr, ys = [], [], []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(healthycat) * 5, 9), dtype=np.float32)
    for jj in range(len(healthycat)):
        valloc = list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc) * ti

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(healthycatfin, ys)
    y_pred = clf.predict(X_test)
    print("ok")

    print("\n[MODEL PREDICTION]")
    print("  Predicted Labels:", list(y_pred))

    global breakfast_suggestions, lunch_suggestions, dinner_suggestions
    breakfast_suggestions = []
    lunch_suggestions = []
    dinner_suggestions = []
    max_items = min(len(y_pred), len(Food_itemsdata))
    for ii in range(max_items):
        if y_pred[ii] == 2:
            food_item = Food_itemsdata.iloc[ii]
            if int(veg) == 1 and int(data["VegNovVeg"].iloc[ii]) != 0:
                print("Skipping non-veg item:", food_item)
                continue
            if ii in breakfastfoodseparatedID:
                breakfast_suggestions.append(food_item)
                print("  - Breakfast:", food_item)
            else:
                if ii in LunchfoodseparatedID:
                    lunch_suggestions.append(food_item)
                    print("  - Lunch:", food_item)
                if ii in DinnerfoodseparatedID:
                    dinner_suggestions.append(food_item)
                    print("  - Dinner:", food_item)
    if not (breakfast_suggestions or lunch_suggestions or dinner_suggestions):
        print("  No suitable food items found.")
    print("===========================================\n")
########################################weight gain###########################

def Weight_Gain():
    print(" Age: %s\n Veg-NonVeg: %s\n Weight: %s\n Height: %s\n" % (e1.get(), e2.get(), e3.get(), e4.get()))
    print("\n===== DIET RECOMMENDATION: Weight Gain =====")
    print("User Input:")
    print(f"  Age       : {e1.get()}")
    print(f"  Veg/NonVeg: {e2.get()}")
    print(f"  Weight    : {e3.get()}")
    print(f"  Height    : {e4.get()}")

    data = pd.read_csv('input.csv')
    data.head(5)
    Food_itemsdata = data["Food_items"]
    Breakfastdata = data["Breakfast"]
    Lunchdata = data["Lunch"]
    Dinnerdata = data["Dinner"]

    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []
    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if int(Breakfastdata.iloc[i]) == 1:
            breakfastfoodseparated.append(Food_itemsdata.iloc[i])
            breakfastfoodseparatedID.append(i)
        if int(Lunchdata.iloc[i]) == 1:
            Lunchfoodseparated.append(Food_itemsdata.iloc[i])
            LunchfoodseparatedID.append(i)
        if int(Dinnerdata.iloc[i]) == 1:
            Dinnerfoodseparated.append(Food_itemsdata.iloc[i])
            DinnerfoodseparatedID.append(i)

    print("BREAKFAST FOOD ITEMS:", breakfastfoodseparated)
    print("LUNCH FOOD ITEMS:", Lunchfoodseparated)
    print("DINNER FOOD ITEMS:", Dinnerfoodseparated)
    print("\n[MEALS]")
    print("  Breakfast:", breakfastfoodseparated)
    print("  Lunch    :", Lunchfoodseparated)
    print("  Dinner   :", Dinnerfoodseparated)

    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID].T.iloc[Valapnd].T
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID].T.iloc[Valapnd].T
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID].T.iloc[Valapnd].T

    age = int(e1.get())
    veg = float(e2.get())
    weight = float(e3.get())
    height = float(e4.get())
    bmi = weight / (height ** 2)
    print(f"\n[INFO] BMI Calculated: {bmi:.2f}")

    if bmi < 16:
        clbmi = 4
    elif bmi < 18.5:
        clbmi = 3
    elif bmi < 25:
        clbmi = 2
    elif bmi < 30:
        clbmi = 1
    else:
        clbmi = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp + 20)
        for i in test_list:
            if i == age:
                print("Age is between", str(lp), "and", str(lp+10))
                agecl = round(lp / 20)

    print("Your BMI is:", bmi)

    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi + agecl) / 2
    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)

    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dnrlbl = kmeans.labels_

    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    lnchlbl = kmeans.labels_

    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    brklbl = kmeans.labels_

    print("\n[CLUSTERING RESULTS]")
    print("  Dinner Clusters:", list(dnrlbl))
    print("  Lunch Clusters :", list(lnchlbl))
    print("  Breakfast Clusters:", list(brklbl))

    datafin = pd.read_csv('inputfin.csv')
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]].T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]].T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]].T

    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()

    weightlosscat = weightlosscatDdata[1:, :]
    weightgaincat = weightgaincatDdata[1:, :]
    healthycat = healthycatDdata[1:, :]

    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)

    t = r = s = 0
    yt, yr, ys = [], [], []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightgaincat), 10), dtype=np.float32)
    for jj in range(len(weightgaincat)):
        valloc = list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc) * ti

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(weightgainfin, yr)
    y_pred = clf.predict(X_test)
    print("ok")

    print("\n[MODEL PREDICTION]")
    print("  Predicted Labels:", list(y_pred))

    global breakfast_suggestions, lunch_suggestions, dinner_suggestions
    breakfast_suggestions = []
    lunch_suggestions = []
    dinner_suggestions = []
    max_items = min(len(y_pred), len(Food_itemsdata))
    for ii in range(max_items):
        if y_pred[ii] == 2:
            food_item = Food_itemsdata.iloc[ii]
            if int(veg) == 1 and int(data["VegNovVeg"].iloc[ii]) != 0:
                print("Skipping non-veg item:", food_item)
                continue
            if ii in breakfastfoodseparatedID:
                breakfast_suggestions.append(food_item)
                print("  - Breakfast:", food_item)
            else:
                if ii in LunchfoodseparatedID:
                    lunch_suggestions.append(food_item)
                    print("  - Lunch:", food_item)
                if ii in DinnerfoodseparatedID:
                    dinner_suggestions.append(food_item)
                    print("  - Dinner:", food_item)
    if not (breakfast_suggestions or lunch_suggestions or dinner_suggestions):
        print("  No suitable food items found.")
    print("===========================================\n")

################################################################################
# PDF Generation Function
################################################################################
def generate_pdf():
    """
    Generate a PDF with three columns: Breakfast, Lunch, and Dinner suggestions.
    """
    global breakfast_suggestions, lunch_suggestions, dinner_suggestions
    file_name = "Suggested_Food_Items.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter

    # Draw header.
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Diet Recommendation System")
    c.setFont("Helvetica", 12)

    # Column headers.
    breakfast_x = 50
    lunch_x = 220
    dinner_x = 390
    header_y = height - 80
    c.drawString(breakfast_x, header_y, "BREAKFAST SUGGESTIONS:")
    c.drawString(lunch_x, header_y, "LUNCH SUGGESTIONS:")
    c.drawString(dinner_x, header_y, "DINNER SUGGESTIONS:")

    # Starting y positions and line spacing.
    breakfast_y = header_y - 20
    lunch_y = header_y - 20
    dinner_y = header_y - 20
    line_height = 15

    # Print breakfast suggestions.
    for item in breakfast_suggestions:
        c.drawString(breakfast_x, breakfast_y, f"- {item}")
        breakfast_y -= line_height
        if breakfast_y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            breakfast_y = height - 50

    # Print lunch suggestions.
    for item in lunch_suggestions:
        c.drawString(lunch_x, lunch_y, f"- {item}")
        lunch_y -= line_height
        if lunch_y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            lunch_y = height - 50

    # Print dinner suggestions.
    for item in dinner_suggestions:
        c.drawString(dinner_x, dinner_y, f"- {item}")
        dinner_y -= line_height
        if dinner_y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            dinner_y = height - 50

    c.save()
    messagebox.showinfo("Success", f"PDF saved as {file_name}")

################################################################################
# Main UI
################################################################################
Label(main_win, text="Age").grid(row=0, column=0, sticky=W, pady=4)
Label(main_win, text="Veg/NonVeg\n(1 for veg, 0 for nonveg)").grid(row=1, column=0, sticky=W, pady=4)
Label(main_win, text="Weight").grid(row=2, column=0, sticky=W, pady=4)
Label(main_win, text="Height").grid(row=3, column=0, sticky=W, pady=4)

e1 = Entry(main_win)
e2 = Entry(main_win)
e3 = Entry(main_win)
e4 = Entry(main_win)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)

Button(main_win, text='Quit', command=main_win.quit).grid(row=5, column=0, sticky=W, pady=4)
Button(main_win, text='Weight Loss', command=Weight_Loss).grid(row=1, column=4, sticky=W, pady=4)
Button(main_win, text='Weight Gain', command=Weight_Gain).grid(row=2, column=4, sticky=W, pady=4)
Button(main_win, text='Healthy', command=Healthy).grid(row=3, column=4, sticky=W, pady=4)
Button(main_win, text="Download Suggested Items PDF", command=generate_pdf)\
    .grid(row=4, column=4, sticky=W, pady=4)

main_win.geometry("400x200")
main_win.wm_title("DIET RECOMMENDATION SYSTEM")
main_win.mainloop()
