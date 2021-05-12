import os
import re
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from typing import Sequence

# spark-submit --driver-class-path postgresql-42.2.18.jar --executor-memory 8g --driver-memory 5g Zone_Candidate_Col.py > test_output.txt


df = pd.read_csv('City_Zoning_Attributes.csv', index_col='Unnamed: 0')

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

df_spark = ss.createDataFrame(df)
df_spark = df_spark.withColumnRenamed('Line No.', 'Line')


def zone_Candidates_Spark(line_no, policy, city) -> list:
    """
    Searches in the window of lines for any zone candidates to which an attribute may be relevant.

    Returns: list which is then appended as a new column to the inputted dataframe
    """
    zone_column = []
    context_window = 20
    #     line_no = x.select("Line No.").head()[0]
    #     policy = x.select("Policy Subsection").head()[0]
    #     city = x.select("City").head()[0]
    #         line_no = df.loc[row]['Line No.']
    #         policy = df.loc[row]['Policy Subsection']
    #         city = df.loc[row]['City']

    root = '/Users/okeefe/Box/USF Data Science Practicum/2020-21/Okeefe/Project_1_Policy_Parsing/City_Policies'

    key_words = [policy, city]

    all_files = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            all_files.append(os.path.join(path, name))

    relevant_paths = [path for path in all_files if all(word in path for word in key_words)]

    # hit = df.loc[row]['Line No.'] - 1
    hit = line_no - 1

    path = relevant_paths[0]
    extract_zones = []

    with open(path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if hit - context_window < i < hit + context_window:
            pot_zone = re.findall(r'[A-Z]+\d-*\d*[A-Z]*', line)
            extract_zones.extend(pot_zone)

    if len(extract_zones) > 0:
        zones_found = ', '.join(val for val in list(set(extract_zones)))
    else:
        zones_found = np.nan

    return zones_found


zone_Candidates_udf = udf(zone_Candidates_Spark, StringType())

df_final = df_spark.withColumn('Zone_Candidates',
                               zone_Candidates_udf(df_spark['Line'],
                                                   df_spark['Policy Subsection'],
                                                   df_spark['City']))

df_working = df_final.toPandas()
df_working.to_csv('City_Zoning_Attributes_with_Zones_Final.csv')