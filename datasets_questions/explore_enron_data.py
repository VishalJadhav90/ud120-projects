#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

count = 0
for person, entry in enron_data.items():
    '''
    if "Lay Kenneth".upper() in person:
        print(entry)
    if "Skilling Jeffrey".upper() in person:
        print(entry)
    if "Fastow Andrew".upper() in person:
        print(entry)
    '''
    if entry['total_payments'] == 'NaN':
        count = count + 1
print(count)
print(len(enron_data))