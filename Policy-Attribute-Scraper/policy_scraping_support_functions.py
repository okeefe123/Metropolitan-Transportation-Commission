import os
import sys
import re
# import nltk
# from nltk.tokenize import word_tokenize
import pandas as pd
# from word2number import w2n
from typing import Tuple

from utils_io import *

bucket = 'mtc-redshift-upload'
socrata_data_id = 'qdrp-c5ra'

whitelist = ['zoning', 'buildings', 'housing', 'building', 'development', 'land']

attr_regex_dict = {'max_far': [r'(?i)floor.+area.+ratio', r'(?i)max.+far'],
                   'max_dua': [r'(?i)dua', r'(?i)dwelling.+unit.+acre', r'(?i)lots.+acres', r'(?i)acres.+dwelling.+unit', r'(?i)dwelling.+(acre|unit)', r'(?i)unit.+acres?'],
                   'building_height' : [r'(?i)building.+height', r'(?i)maximum.+height', r'(?i)height.+structures?'],
                   'units_per_lot' : [r'(?i)units?.+per.+lot', r'(?i)units?.+sq(are)*.*fe*t'],
                   'minimum_lot_sqft': [r'(?i)unit.+per.+fe*t', '(?i)mini?m?u?m?.+lot.+sqa?r?e?.+fe*t', r'(?i)sq(uare)*.+fo*e*t', r'(?i)gross.+lot.+area']
                  }

attr_range_dict = {
             'max_far':(0, 5),
             'max_dua': (0, 100),
             'building_height' : (15, 250),
             'units_per_lot' : (0.5, 5),
             'minimum_lot_sqft': (600, 1e8) 
              }

#############CLASSIFIER PIPELINE (SINGLE CITY)#######################
def city_code_table(city_name=None) -> pd.DataFrame:
    """
    Pulls city zoning information from socrata table 'Regional Zoning Codes 2018' and sets
    the index of the pandas dataframe as the zone codes
    
    Returns dataframe of the desired city.
    """


    query_args = {'select': '*', #  'RecID, zn_code, max_far, max_dua, building_height, units_per_lot, minimum_lot_sqft',
                  'where': f"city_name='{city_name}'"
                 }
    if city_name == None:
        del query_args['where']
        
    regional_zoning_codes_old = pull_df_from_socrata(socrata_data_id, query_args=query_args)

    for attr in attr_regex_dict.keys():  # Resolves 
        if attr not in regional_zoning_codes_old.columns:
            regional_zoning_codes_old[attr] = np.nan


    regional_zoning_codes_old.set_index('zn_code', inplace=True)
    return regional_zoning_codes_old


def zone_context(contexts: dict, zoning_codes: list) -> dict:
    """
    DEPRECATED (zone parsing did not produce desired results)
    
    Takes in the dictionary data structure and further parses the information by zone
    Returns:
    
    Key: (zone, attribute)
    Value: list of (value_found, context, policy_section) for the key
            List type is important!! Ideally, there is only one tuple per key, so a list implies redundant information. 
            NOTE: one of the zone keys is "no_zone". This is reserved for all attribute values that were not found with an associated zoning codes from the original socrata table.
    """
    
    goods = dict()

    for zone in zoning_codes:
        for target_attr in attr_regex_dict.keys():
            #print("#########################Target Attribute:", target_attr)
            for context in contexts[target_attr]:
                path = context[0]
                lines = context[1]
                zone_attr_vals = []
                no_zone_vals = []
                
                attr_range = attr_range_dict[target_attr]  # Minimum/Maximum range of realistic values for attributes
                for line in lines:
                    print(line)
                    items = find_the_values(line, attr_range[0], attr_range[1])
                    if len(items) > 0:
                        #print("This is the found context:", line, "###############################")
                        #print()
                        #print(items)
                        #print()
                        val = items[0][0]
                        context = items[0][1]
                        if zone in line:
                            zone_attr_vals.append((val, context, path))
                        else:
                            no_zone_vals.append((val, context, path))
                            #print(no_zone_vals)
                
                goods[(zone, target_attr)] = zone_attr_vals
                goods[('no_zone', target_attr)] = no_zone_vals
#     try:
#         print(goods['no_zone'])
#     except:
#         print("nothing found")
#         pass

    return goods


def update_city_table(old_df: pd.DataFrame, goods: dict) -> pd.DataFrame:
    """Updates a table based on city policies scraped from various jurisdications"""
    new_df = old_df.copy()
    updates = 0
    for key, val in goods.items():
        zone = key[0]
        attribute = key[1]
        if len(val)>0:
            if zone != 'no_zone':
                new_df.at[zone,attribute] = list(val[0][0])[0]
                updates += 1
    
    return new_df, updates

# Helper functions for the level_one_classifier:

def open_file_by_key(bucket: str, key:str) -> list:
    """Returns file from bucket as an array of lines"""
    
    s3 = boto3.client('s3')
    lines = s3.get_object(Bucket=bucket, Key = key)['Body'].iter_lines()
    lines_split = [str(line, 'utf-8', 'ignore').replace('\xa0', ' ') for line in lines]  # formatting unicode weirdness
    return lines_split

def normalize_text(path: str) -> list:
    """Returns a list of tokenized/normalized words, digits, dates from a string"""
    
    # strip punctuation
    path = re.sub(r'[/.,!?;]', ' ', path)  
    # reformat the title such that it's int the format: 'Title-Number'
    # Assumes only one instance of title in link
    titleno = re.findall(r'[Tt]itle [0-9A-Z]+', path)
    chapterno = re.findall(r'[Cc]apter [0-9A-Z]+', path)
    # Workaround for empty-list error
    if len(titleno) > 0:
        titleno = titleno[0].replace(' ', '-')
        path = re.sub(r'[Tt]itle [0-9A-Z]+', titleno, path)
    if len(chapterno) > 0:
        chapterno = titleno[0].replace(' ', '-')
        path = re.sub(r'[Tt]itle [0-9A-Z]+', chapterno, path)
    # tokenize the formatted text
    tokenized = nltk.word_tokenize(path)
    tokenized = [ch.lower() for ch in tokenized]
    tokenize = ' '.join(tokenized)
    return tokenized

def most_Recent_Policy(city_policies: set) -> set:
    """
    Helper function for get_policies() which only pulls most recent policies  
    """
    
    found = {}

    for policy in city_policies:
            pieces = policy.split("/")
            date = pieces[2]
            title = pieces[3]
            if title in found.keys():
                new_time = datetime.strptime(date, "%m-%d-%y")
                old_time = datetime.strptime(found[title], "%m-%d-%y")
                if new_time > old_time:
                    found[title] = date
            else:
                found[title] = date

    
    kjafshar = list(city_policies)[0].split("/")[0]
    city = list(city_policies)[0].split("/")[1]

    new_policies = []
    for title, date in found.items():
        new_policies.append("/".join([kjafshar, city, date, title]))

    return set(new_policies)

def get_policies(whitelist=None, city=None, local=True, most_recent=False) -> set:
    """
    Returns a set of all relevant subpolicy paths, using whitelist as a filtering tool for relevant policy subsections
    
    Example use cases:
        - default: all policies in the local subdirectory PATH:                       get_policies()
        - all policies in s3 subdirectories:                                          get_policies(local=False)
        - all relevant policy subsections for "berkeley" in local subdirectory PATH:  get_policies(whitelist=whitelist, city="berkeley")
        - all relevant policy subsections for "berkeley" in s3:                       get_policies(whitelist=whitelist, city="berkeley", local=False)
        - most recent scraped policies for "berkeley" in s3:                          get_policies(whitelist=whitelist, city="berkeley", local=False, most_recent=True)
    """
    
    # Gather all local policies either from local or s3 bucket
    if local:
        PATH = 'City_Policies'
        keys = glob.glob(PATH + '/**/*.txt', recursive=True)
        
    else:
        list_key_args = {'bucket': 'mtc-redshift-upload', 'Prefix':f'test_kjafshar/'}
        keys = list_s3_keys(**list_key_args)
        
    # Filter keys via whitelist
    if whitelist != None:
        keys = {match for match in keys if any(term.lower() in normalize_text(match) for term in whitelist)}
    
    # Filter for specific city if given
    if city != None:
        keys = {match for match in keys if city == match.split("/")[1]}
        
    # Most recent documents if most_recent is switched to True
    if most_recent:
        keys = most_Recent_Policy(keys)
    
    return keys



def print_lines(relevant_info: dict, key_words: list):
    """Prints either the dictionary of all relevant policy lines or a specific city's lines"""
    
    print(f"Key words used in search: {', '.join(word for word in key_words)}")
    print()
    
    if isinstance(relevant_info, dict):
        for key, val in relevant_info.items():
            print("This is the area:", key)
            print()
            print("These are the relevant lines:")

            for num, line in enumerate(val):
                print(f"line #{num+1}:")
                print(line)
            print()
    else:
        print("These are the relevant lines")
        for num, line in enumerate(relevant_info):
            print(f"line #{num+1}:")
            print(line)
        print()
        
        

################EXTRACTING NUMBERS AND CONTEXT IN WHICH THEY WERE FOUND###################################
def find_the_values(line: str, min_val: int, max_val: int) -> list:
    """
    Searches through supplied line, extracting the numbers which are within the range of min_val and
    max_val.
    
    Returns: List of tuples, where index[:][0] are the extracted numbers and index[:][1] is the context from
             which they were extracted.
    """
    
    relevant = []
    
    num_line = numerical_numbers(line)
    num_line = num_line.replace(',', '')
    if any(char.isdigit() for char in num_line):
        nums = re.findall(r'\s[0-9]+\s', num_line)  # excludes policy sections ex: 17.172.050
        parentheses = re.findall(r'\([0-9]+\)', num_line) # nums inside parentheses
        nums.extend(parentheses)
        nums = [re.sub(r'(\(|\)| )', '', x) for x in nums]  # drop all values and convert them into 
        nums = [float(x) for x in nums if float(x) > min_val and float(x) < max_val]
        if len(nums) > 0:
            relevant.append((set(nums), line))

    return relevant


# find_the_values() helper functions:

def numerical_numbers(line: str) -> str:
    """Converts all worded numbers into actual numbers"""
    line = line.split()
    new_val = []
    for val in line:
        try:
            val = str(w2n.word_to_num(val))
            new_val.append(val)
        except:
            new_val.append(val)

    line = ' '.join(new_val)
    return line


#########################CODE EFFICIENCY METRICS##############################

def extraction_efficiency(city_dict: dict, min_target_val, max_target_val) -> None:
    """
    Prints the extracted information, its context, and the efficiency of the search engine for all cities.
    
    Metric: Ratio of number of cities where potentially relevant information was found compared to the total
            number of cities in the inputted dictionary (as a percentage).
    """
    
    heightcount = 0
    for city in city_dict.keys():
        print("city:", city)
        potential_hit = find_the_values(city_dict[city], min_target_val, max_target_val)
        if len(potential_hit) > 0:
            for hit in potential_hit:
                print("value/context:", hit)
            #print("context:", potential_hit[1])
            print()
            heightcount += 1

    print("% found:", heightcount / len(city_dict.keys())*100)
    
    
    
    
####################DISTRICTS ANALYSIS###########################
def get_districts(city: str) -> Tuple[set, list]:
    """
    Identifies all districts within a city.
    
    Returns the districts as well as the context in which they were found.
    """
    
    keys = get_policies(city)
    bucket = 'mtc-redshift-upload'
    district_list = []
    district_context = []
    
    for key in keys:
        lines = open_file_by_key(bucket, key)
        for line in lines:
            if bool(re.search(r'[A-Z]-\d[A-Z]?', line)):
                districts = re.findall(r'[A-Z]-\d[A-Z]?', line)
                district_list.extend(districts)
                district_context.append(line)
    
    return set(district_list), district_context


def get_cities() -> set:
    """Parses the path list from s3 bucket and returns names of all scraped cities (unformatted)"""
    
    paths = get_policies(all_policies=True)
    cities = set()
    
    for path in paths:
        city = path.split('/')[1]
        cities.add(city)

    return cities


def get_city_policies(city: str) -> set:
    """Return all policies scraped from an inputted city"""
    full_list = get_policies(all_policies=True)
    city_list = set()
    
    for entry in full_list:
        if city == entry.split('/')[1].lower():
            city_list.add(entry)
            
    return city_list





####################### Deprecated Functions ##########################
# def organize_by_attribute(attr_regex_dict: dict, rel_policy_path: set, local=False) -> dict:
#     """
#     Opens the desired policies via s3 bucket path and filters by attribute.
#     Optional "local" flag pulls the policy data from local directory (debugging)
#     Returns:
    
#     Key: attribute
#     Value: tuple (path: str, relevant_lines: set)
#     """
#     relevant_info={}

    
#     for attr in attr_regex_dict.keys():
#         relevant_info[attr] = list()
#         attr_range = attr_range_dict[attr]

#         for path in rel_policy_path:
#             policy_section = path.split('/')[-1].partition('.')[0]
#             if local: #If local parameter is true, override relative_policy_path with the local file paths
#                 path = 'City_Policies/' + "/".join(path.split("/")[2:])
#                 data = read_text('City_Policies/alameda/01-23-20/CHAPTER XII - DESIGNATED PARKING.txt').split('\n')
             
#             else:
#                 data = open_file_by_key(bucket, path)
            
#             regex_list = attr_regex_dict[attr]
#             relevant_lines = list()
            
#             for exp in regex_list:
#                 for line in data:
#                     if bool(re.search(exp, line)):
#                         #relevant_lines.add(line)   # Eventually replace with a more in-depth function extraction of value/line
#                         items = find_the_values(line, attr_range[0], attr_range[1])
                        
#                         if len(items) > 0:
#                             val = items[0][0]
#                             context = itemss[0][1]
#                             relevant_lines.add((val, context))
            
#             relevant_info[attr].append((policy_section, relevant_lines))
    
#     for key, info in relevant_info.items():
#         print(key)
#         print(info)
#     return relevant_info


# def organize_by_attribute(city: str, attr_regex_dict: dict, rel_policy_path: set, local=False) -> dict:
#     """
#     Opens the desired policies via s3 bucket path and filters by attribute.
#     Optional "local" flag pulls the policy data from local directory (debugging)
#     Returns:
    
#     Key: attribute
#     Value: tuple (path: str, relevant_lines: set)
#     """
#     relevant_info={}

    
#     for attr in attr_regex_dict.keys():
#         relevant_info[attr] = list()
#         attr_range = attr_range_dict[attr]

#         for path in rel_policy_path:
#             policy_section = path.split('/')[-1].partition('.')[0]
#             if local: #If local parameter is true, override relative_policy_path with the local file paths
#                 path = 'City_Policies/' + city.lower()+ "/"+ "/".join(path.split("/")[2:])
#                 data = read_text(path).split('\n')
             
#             else:
#                 data = open_file_by_key(bucket, path)
            
#             regex_list = attr_regex_dict[attr]
#             relevant_lines = list()
            
#             for exp in regex_list:
#                 for num, line in enumerate(data):
#                     if bool(re.search(exp, line)):
#                         #relevant_lines.add(line)   # Eventually replace with a more in-depth function extraction of value/line
#                         items = find_the_values(line, attr_range[0], attr_range[1])
#                         fraction = num / len(data) * 100
#                         if len(items) > 0:
#                             #print("HIT!!!! What did we find?####################################################")
#                             entry = items[0]
#                             vals = entry[0]
#                             context = entry[1]
#                             #print("vals:", vals)
#                             #print("context:", context)
#                             #print(val)
#                             #for context in items:
#                             #    print(context[1])
#                             relevant_lines.append((policy_section, list(vals), context, num+1, fraction))
            
#             relevant_info[attr].append(relevant_lines)
#         #print()
# #     for key, info in relevant_info.items():
# #         print(key)
# #         print(info)
#     return relevant_info

