from collections import defaultdict

import numpy as np
import pandas as pd

import array
import cPickle as pickle
import os
import datetime

if __name__ == '__main__':
    if (not os.path.exists('good_data.csv')):
        
        data = pd.read_csv('Outbreak_240817.csv')
        data = data.drop(columns=['source', 'latitude', 'longitude', 'country', 'admin1', 'localityName', 'localityQuality', 'reportingDate', 'status', 'serotypes', 'speciesDescription', 'sumAtRisk', 'sumDeaths', 'sumDestroyed', 'sumSlaughtered', 'humansGenderDesc', 'humansAge', 'humansAffected', 'humansDeaths'])
        
        data = data.dropna()
        data = data.reset_index()
        data = data.drop(columns=['index'])
        
        print data.head()
        for i in range(len(data.observationDate)):
            date = data.loc[i,'observationDate']
            if (isinstance(date, str)):
                   data.loc[i,'observationDate'] = datetime.datetime.strptime(date, "%d/%m/%Y")
        print(data.observationDate[1])
        data.to_csv('good_data.csv')
    else:
         data = pd.read_csv('good_data.csv')
         data[['observationDate']] = data[['observationDate']].apply(pd.to_datetime)

    #most_common_diseases = data.disease.index
    all_diseases = data.disease.value_counts().index

    all_countries = data.region.value_counts().index
    
    for disease in all_diseases[0:1]:
        print(disease)
        one_disease_table = data.loc[data['disease'] == disease]
        #for i in one_disease_table:
         #   print type(i)
        for country in all_countries[0:1]:
            print country
            one_country_table = one_disease_table.loc[one_disease_table['region'] == country]
            if len(one_country_table)!=0:
                date_min = min(one_country_table.observationDate)
                date_max = max(one_country_table.observationDate)
                for year in range(date_min.year,date_max.year+1):
                    for month in range(date_min.month,date_max.month+1):
                        number_cases = 0
                        if (len(one_country_table)!=0):
                            print one_country_table.observationDate.head(1)
                            print datetime.datetime(year, month%12 +1 , 1)-datetime.timedelta(days=1)
                            delta = (one_country_table.observationDate.head(1) - datetime.datetime(year, (month-1)%12 +1 , 1)-datetime.timedelta(days=1))
                            print delta
                        case_in_month = one_country_table.loc[(one_country_table['observationDate'] - datetime.datetime(year, (month-1)%12 +1 , 1)-datetime.timedelta(days=1)).days <= 31]
                       
                        if len(case_in_month) != 0:
                            number_cases = case_in_month.sumCases.sum()
                            #print number_cases
            #processed_data = 0 # les datas une fois bien rangee et tout
