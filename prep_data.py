from collections import defaultdict

import numpy as np
import pandas as pd

import array
import cPickle as pickle
import os
import datetime

if __name__ == '__main__':

    data = pd.read_csv('Outbreak_240817.csv')
    data = data.drop(columns=['source', 'latitude', 'longitude', 'country', 'admin1', 'localityName', 'localityQuality', 'reportingDate', 'status', 'serotypes', 'speciesDescription', 'sumAtRisk', 'sumDeaths', 'sumDestroyed', 'sumSlaughtered', 'humansGenderDesc', 'humansAge', 'humansAffected', 'humansDeaths'])
    
    data = data.dropna()
    data = data.reset_index()
    data = data.drop(columns=['index'])

    first_date = datetime.datetime.strptime(data.loc[0,'observationDate'], "%d/%m/%Y")
    data_process = pd.DataFrame({'region':data.loc[0,'region'],
                                 'observationDateYear': first_date.year,
                                 'observationDateMonth': first_date.month,
                                 'disease': data.loc[0,'disease'],
                                 'sumCases': data.loc[0,'sumCases']},index=[0])
    
    for i in range(1,len(data.observationDate)):
        date = datetime.datetime.strptime(data.loc[i,'observationDate'], "%d/%m/%Y")
        data_process = data_process.append(pd.DataFrame({'region':data.loc[i,'region'],
                                                             'observationDateYear': date.year,
                                                             'observationDateMonth': date.month,
                                                             'disease': data.loc[i,'disease'],
                                                             'sumCases': data.loc[i,'sumCases']},index=[i]), ignore_index = True)
        

    dict_country = dict()
    dict_disease = dict()
    #most_common_diseases = data.disease.index
    all_diseases = data_process.disease.value_counts().index
    index = 1
    for disease in all_diseases:
        dict_disease[disease] = index
        index = index + 1

    all_countries = data_process.region.value_counts().index
    index = 1
    for country in all_countries:
        dict_country[country] = index
        index = index + 1
        
    final_data =  pd.DataFrame()
    
    for disease in all_diseases:
        one_disease_table = data_process.loc[data_process['disease'] == disease]
        for country in all_countries:
            one_country_table = one_disease_table.loc[one_disease_table['region'] == country]
            if len(one_country_table)!=0:
                year_min = min(one_country_table.observationDateYear)
                month_min = min(one_country_table.loc[one_country_table['observationDateYear']==year_min].observationDateMonth)
                year_max = max(one_country_table.observationDateYear)
                month_max = max(one_country_table.loc[one_country_table['observationDateYear']==year_max].observationDateMonth)
                for year in range(year_min,year_max+1):
                    for month in range(month_min,month_max+1):
                        if (len(one_country_table)!=0):
                            case_in_month = one_country_table.loc[(one_country_table['observationDateYear']==year) & (one_country_table['observationDateMonth']==month)]
                            
                            if len(case_in_month) != 0:        
                                number_cases = case_in_month.sumCases.sum()
                                
                                final_data = final_data.append( pd.DataFrame({'region':dict_country[country],
                                                                              'observationDateYear': year,
                                                                              'observationDateMonth': month,
                                                                              'disease': dict_disease[disease],
                                                                              'sumCases': number_cases},index=[len(final_data)]), ignore_index = True)



    final_data = final_data.sample(frac=1)
    
    label = final_data.ix[:,4]
    train_data = final_data.ix[:,:4]
    test_X = []

    train_X = train_data[:140]
    train_y = label[:140]
    valid_X = train_data[140:]
    valid_y = label[140:]
    
    data_for_MLP = (
        (train_X, train_y),
        (valid_X, valid_y),
        (test_X,),
    )
    data_file = "preprocessed_data.pkl"
    print "...saving data pickle: %s" % data_file
    with open(data_file, 'wb') as f:
        pickle.dump(data_for_MLP, f)
