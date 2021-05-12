import numpy as np
import pandas as pd
from collections import defaultdict
import math

import sys
# local import
sys.path.insert(0, '../Project_1_Policy_Parsing')
from utils_io import *
import random
import time
import warnings
warnings.filterwarnings('ignore')


def main(event=None, context=None):
    start = time.time()
    socrata_data_id = event['socrata_data_id']
    df_master = pull_df_from_socrata(socrata_data_id)
    #df_updated = use_cases(df_master)
    juris_entries = pd.DataFrame(event['jurisdiction_entries'])
    updates = CheckUpdates(df_master, juris_entries, primary_key='recid', parcels=False)
    updates.changed_rows()
    updates.update_analysis()
    print(updates.warning_table)
    print(time.time()-start)
    print("The end")

def pretty_print(df):
    """Can print dataframes whenever""" 
    with pd.option_context('display.max_colwidth', None, 'display.max_rows', None,):
        display(df)
        
def load_data():
    """Load data from socrata id"""
    socrata_data_id = 'qdrp-c5ra'
    df_old = pull_df_from_socrata(socrata_data_id)
    
    return df_old

class CheckUpdates:
    def __init__(self, df_old, df_updated, primary_key, parcels=False, compare_masters=False):
        self.compare_masters = compare_masters
        self.df_old = df_old
        
        # second index will be a dataframe
        if compare_masters:
            self.df_updated = df_updated
        # Reformat json upload such that edit dates are in datetime format
        # WARNING!! Testing json files needed to be adjusted by 10^3 before converting from UNIX epoch to date/time. This may not be the case and could
        # be a future source of bugs.
        else:
            self.df_updated = df_updated
            self.df_updated['edit_date'] = pd.to_datetime(self.df_updated['edit_date'].map(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000))))
            
        self.primary_key = primary_key
        self.row_comparison = pd.DataFrame(columns = self.df_updated.columns)
        self.update_log = pd.DataFrame(columns = [str(self.primary_key),
                                                  'city_name',
                                                  'cols_updated',
                                                  'updated_from_NaN',
                                                  'changed_vals',
                                                  'out_of_range',
                                                  'warnings',
                                                  'editor',
                                                  'edit_timestamp',
                                                  'edit_type (C/U/D)',
                                                  'main_version',
                                                  'old_vals',
                                                  'new_vals'
                                                 ]
                                       )
        self.reasonable_float_reference()
        self.changed_rows()
        self.organize_coltypes()
        self.parcels = parcels
        if self.parcels:
            self.parcel_df = pull_df_from_redshift_sql('select * from policy.zoning_parcels_18')


    def changed_rows(self):
        """Creates a matrix containing all changed rows (updated listed first)"""
        
        # If comparing master versions, just keep changed/added/removed rows when comparing the two
        if self.compare_masters == True:
            changed_rows = pd.concat([self.df_updated, self.df_old]).drop_duplicates(keep=False)
        # otherwise, identify the rows in the old dataframe with the same key as those updated
        else:
            rel_keys = self.df_updated['recid']
            old_relevant = self.df_old[self.df_old['recid'].isin(rel_keys)]
            changed_rows = self.df_updated.append(old_relevant)
            
        self.row_comparison = changed_rows.sort_values(["city_name", self.primary_key])

    def organize_coltypes(self):
        """Using the self.row_comparison table, categorize the inputted columns into float and object"""
        num_cols = []
        for colname, colbool in (self.row_comparison.dtypes == float).items():
            if colbool == True:
                num_cols.append(colname)

        if self.primary_key in num_cols: num_cols.remove(self.primary_key)
        self.num_cols = num_cols

        str_cols = []
        for colname, colbool in (self.row_comparison.dtypes == object).items():
            if colbool == True:
                str_cols.append(colname)

        if self.primary_key in str_cols: str_cols.remove(self.primary_key)
        self.str_cols = str_cols

    def update_analysis(self):
        """
        MAIN ANALYSIS FUNCTION
        
        Construction of update log which keeps track of the details for added/dropped/changed rows
        """
        self.row_comparison.sort_values('city_name')
        #print(self.row_comparison)
        keys = self.row_comparison[self.primary_key].unique()

        #Less elegant way of constructing dataframe. These attributes will be
        #tacked on at the very end to preserve list data structure in df cells
        cols_updated_list = []
        warnings_list = []
        old_vals_list = []
        new_vals_list = []

        # Loop through all the keys (ex: recID for land use key)
        for key in keys:
            # Initialize a dictionary which organizes each of the desired column values
            # key: column name, value: column value
            update_stats = defaultdict()
            
            # primary_key:
            update_stats[str(self.primary_key)] = key

            # collect metadata from the updated entry
            editor, edit_timestamp, main_version = self.pull_metadata(key)
            update_stats['editor'] = editor
            update_stats['edit_timestamp'] = edit_timestamp
            update_stats['main_version'] = main_version

            # get list of columns updated in entry
            column_list = self.cols_updated(key)
            #update_stats['cols_updated'] = str(column_list)

            # collect the city name
            try:
                update_stats['city_name'] = self.df_updated[self.df_updated[str(self.primary_key)] == key]['city_name'].iloc[0]
                cols_updated_list.append(column_list)
            # if not present in updated_df, then can infer that the row was deleted from the main version socrata table during 
            # jurisdiction entry.
            except:
                column_list = 'row deleted'
                cols_updated_list.append(column_list)
                update_stats['city_name'] = self.df_old[self.df_old[str(self.primary_key)] == key]['city_name'].iloc[0]

            # number of values that are updated from NaN
            debug_key = key
            num_nan_updated = self.updated_from_nan(self.num_cols, key)
            update_stats['updated_from_NaN'] = num_nan_updated

            # number of columns where the values were changed (NaN if row was updated/deleted)
            if isinstance(column_list, str):
                update_stats['changed_vals'] = np.nan
            else:
                update_stats['changed_vals'] = len(column_list) - num_nan_updated

            # out of range values based on values found in "master" table logged
            warnings, oor = self.out_of_bounds(key)
            update_stats['out_of_range'] = oor

            # log string entries if it's the first time they've been found
            # again based on distinct string entries in "master" table
            str_warnings = self.check_string_update(key)
            if len(str_warnings) > 0:
                warnings.extend(str_warnings)
            warnings_list.append(warnings)

            # did the jurisdction delete, create, or update a row in their entry?
            if column_list == 'row deleted':
                update_stats['edit_type (C/U/D)'] = 'D'
                old_vals_list.append(self.gather_old_rows(key))
                new_vals_list.append(np.nan)
            elif column_list == 'row created':
                update_stats['edit_type (C/U/D)'] = 'C'
                old_vals_list.append(np.nan)
                new_vals_list.append(self.gather_new_rows(key))
            else:
                update_stats['edit_type (C/U/D)'] = 'U'
                old_vals_list.append(self.gather_old_rows(key))
                new_vals_list.append(self.gather_new_rows(key))

            # create a single row dataframe with the above aggregated information
            stats_df = pd.DataFrame(update_stats, index=[0])

            # append the row to the update log
            self.update_log = self.update_log.append(stats_df)

        # Inelegant work-around to including lists as cells in the dataframe
        self.update_log['cols_updated'] = cols_updated_list
        self.update_log['warnings'] = warnings_list
        self.update_log['new_vals'] = new_vals_list
        self.update_log['old_vals'] = old_vals_list

        # Optionally append column which displays the parcels affected
        # WARNING!! Signficantly longer runtime since parcel table is huge
        if self.parcels:
            self.parcels_affected()

        # Create a table which just gives potential warnings
        self.table_of_warnings()
        

    def parcels_affected(self):
        """Append column to update log which counts how many parcels are affected by jurisdiction entry"""
        
        parcel_count = self.parcel_df.groupby('recid').count().reset_index()
        parcel_count = parcel_count[['recid', 'geom_id']].rename(columns={"geom_id":"num_parcels_affected"})
        self.update_log = self.update_log.merge(parcel_count, on='recid', how='left')

    def table_of_warnings(self):
        """
        Construction of warnings table
        
        Each warning gets its own row (even if they belong to same primary key)
        """
        
        recid_vals = self.update_log['recid'].values.tolist()
        warning_list = self.update_log['warnings'].values.tolist()
        cities = self.update_log['city_name'].values.tolist()
        editors = self.update_log['editor'].values.tolist()


        recid_warn_tuple_list = []

        for recid, warning, city, editor in zip(recid_vals, warning_list, cities, editors):
            recid_warn_tuple_list.extend([(city, editor, recid, warn) for warn in warning])

        self.warning_table = pd.DataFrame(recid_warn_tuple_list, columns=['city', 'editor', 'recid', 'warn'])

        
    def pull_metadata(self, key):
        pair = self.row_comparison[self.row_comparison[self.primary_key] == key]

        if len(pair) == 1:
            editor = pair['editor'].iloc[0]
            edit_timestamp = pair['edit_date'].iloc[0]
            main_version = 1

        elif len(pair) == 2:
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]
            editor = row_updated['editor'].iloc[0]
            edit_timestamp = row_updated['edit_date'].iloc[0]
            main_version = 1

        return editor, edit_timestamp, main_version

    def cols_updated(self, key):
        """
        Returns the names of the columns where values were changed for a given key
        """
        updated_cols = []

        pair = self.row_comparison[self.row_comparison[self.primary_key] == key]

        if len(pair) == 1:
            updated_cols = 'row created'

        if len(pair) == 2:  # if the recid row was updated, there will only be two rows in this column
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]
            row_old = pair[pair['edit_date'] != pair.edit_date.max()]

            for col in pair.columns:
                if str(row_updated[col]) != str(row_old[col]):
                    updated_cols.append(col)

        return updated_cols

    def updated_from_nan(self, num_cols, key):
        """
        Given an old and updated df, counts the number values that were updated from a NaN
        """
        updated_nan = 0
        pair = self.row_comparison[self.row_comparison[self.primary_key] == key]

        if len(pair) == 1:
            updated_nan = np.nan

        if len(pair) == 2:  # if the recid row was updated, there will only be two rows in this column
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]
            row_old = pair[pair['edit_date'] != pair.edit_date.max()]


            new_num_nan = int(row_updated[num_cols].isnull().sum(axis=1))
            old_num_nan = int(row_old[num_cols].isnull().sum(axis=1))
            updated_nan += (old_num_nan - new_num_nan)

        return updated_nan

    
    def reasonable_float_reference(self):
        """
        Identifies all numerical columns and returns a dictionary containing values from %5-95%
        """

        num_list = list(self.df_old.select_dtypes(include=['float']).columns)
        attr_range_dict = {}
        for col in num_list:
            try:
                col_no_na = self.df_old[~self.df_old[col].isnull()][col]
                low = np.percentile(col_no_na, 5)
                high = np.percentile(col_no_na, 95)
                attr_range_dict[col] = (low, high)
            except:
                attr_range_dict[col] = (np.nan, np.nan)

        self.reasonable_vals = attr_range_dict
        
        
    def out_of_bounds(self, key):
        """
        Takes in a row and checks to see if any updated numerical values are outside of the 5%-95% range of already existing values in the table.
        
        returns: list of out-of-range warnings for each row, the number of warnings triggered by a row 
        """
        
        warning_list = []
        pair = self.row_comparison[self.row_comparison[self.primary_key] == key]

        if len(pair) == 1:
            for col in self.num_cols:
                val_u = pair.iloc[0][col]
                if not pd.isna(val_u):
                    if val_u < self.reasonable_vals[col][0] or val_u > self.reasonable_vals[col][1]:
                        warning = f"{col} value is out of range (val = {val_u})"
                        warning_list.append(warning)


        if len(pair) == 2:  # if the recid row was updated, there will only be two rows in this column
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]
            row_old = pair[pair['edit_date'] != pair.edit_date.max()]

            for col in self.num_cols:
                if (float(row_updated[col]) != float(row_old[col])):

                    val_u = row_updated.iloc[0][col]
                    if not pd.isna(val_u):
                        if val_u < self.reasonable_vals[col][0] or val_u > self.reasonable_vals[col][1]:
                            warning = f"{col} value is out of range (val = {val_u})"
                            warning_list.append(warning)

        return warning_list, len(warning_list)


    def check_string_update(self, key):
        """
        Check to see if the string update is unique compared to all existing values for the column in the old dataframe
        """
        str_warnings = []
        pair = self.row_comparison[self.row_comparison[self.primary_key] == key]


        if len(pair) == 1:
            null_cols = {col for col in pair.columns if pair[col].isnull().any()}
            str_cols = set(self.str_cols).difference(null_cols)
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]

            for col in str_cols:
                existing_str = self.df_old[col].unique()
                if pair[col].iloc[0] not in existing_str:
                    str_warnings.append(f"{row_updated[col]} not in other row records for {col}")

        elif len(pair) == 2:
            null_cols = {col for col in pair.columns if pair[col].isnull().any()}
            str_cols = set(self.str_cols).difference(null_cols)
            row_updated = pair[pair['edit_date'] == pair.edit_date.max()]

            for col in str_cols:
                existing_str = self.df_old[col].unique()
                if row_updated[col].iloc[0] not in existing_str:
                    str_warnings.append(f'"{row_updated[col].iloc[0]}" not in other row records for {col}')

        return str_warnings

    # def keys_added_dropped(df_old: pd.DataFrame, df_new: pd.DataFrame, primary_key: str) -> tuple:
    #     """
    #     Given two dataframes, returns the primary keys of columns added and dropped
    #
    #     return: ([dropped keys], [added keys])
    #     """
    #
    #     added = []
    #     dropped = []
    #     merged_df = df_old.merge(right=df_new, how='outer', on=primary_key, suffixes=['', '_'], indicator=True)
    #
    #     if merged_df['_merge'].value_counts()['right_only'] > 0:
    #         added = list(merged_df[merged_df['_merge'] == 'right_only'][primary_key])
    #
    #     if merged_df['_merge'].value_counts()['left_only'] > 0:
    #         dropped = list(merged_df[merged_df['_merge'] == 'left_only'][primary_key])
    #
    #     return added, dropped

    def gather_old_rows(self, key):
        return self.df_old[self.df_old[self.primary_key]==key].values.tolist()[0]

    def gather_new_rows(self, key):
        return self.df_updated[self.df_updated[self.primary_key]==key].values.tolist()[0]


######################### Functions from utils_io: ############################

def create_socrata_client(read_only=True):  # modified for AWS Lambda
    """
    Creates a sodapy Socrata client given credentials
    saved in Application Secure Files Box folder
    """
    # Pre-requisite: ! pip install sodapy
    from sodapy import Socrata
    # socrata_domain = SOCRATA_CREDS['domain']
    # # modified for AWS Lambda:
    socrata_domain = os.environ['socrata_domain']
    # use read-only access creds
    if read_only:
        socrata_user = 'basis'
        # print('connecting to Socrata using read-only credentials')
    # connect with user creds
    else:
        socrata_user = user
    try:
        # username = SOCRATA_CREDS[socrata_user]['username']
        # password = SOCRATA_CREDS[socrata_user]['password']
        # app_token = SOCRATA_CREDS[socrata_user]['app_token']
        # # modified for AWS Lambda:
        username = os.environ['socrata_username']
        password = os.environ['socrata_password']
        app_token = os.environ['socrata_app_token']
        client = Socrata(socrata_domain, username=username,
                         app_token=app_token, password=password, timeout=200)
        return client
    except Exception as e:
        print(e)


def get_nrows_from_socrata(socrata_data_id, read_only=False, query_args=None):
    # get number of rows in the data
    num_rows_args = {'select': 'count(*)'}
    if query_args is not None and 'where' in query_args:
        num_rows_args['where'] = query_args['where']
    client = create_socrata_client()
    num_rows_response = client.get(socrata_data_id, **num_rows_args)
    return int(num_rows_response[0]['count'])


def get_metadata_socrata(socrata_data_id):
    client = create_socrata_client()
    return client.get_metadata(socrata_data_id)


def get_cols_socrata(socrata_data_id):
    metadata = get_metadata_socrata(socrata_data_id)
    return [c['fieldName'] for c in metadata['columns']]


# changed max_chunk_area to 1e5 instead of 1e6 for Lambda
#  (since it was using 129 of 128 MB of memory)
def pull_df_from_socrata(socrata_data_id, max_chunk_area=1e5, read_only=False,
                         query_args=None, ignore_date_types=False):
    """
    Given a data endpoint (e.g. 'fqea-xb6g'), and optionally chunksize,
    read_only, and query_args (dictionary, can only contain select,
    where, order, and limit), returns a pandas DataFrame of the Socrata data.

    query_args notes:
    - See here for Socrata query args: https://dev.socrata.com/docs/queries/
    - Args must be in string format, e.g. {'select': 'joinid, fipco'}

    Look into: currently, specifiying limit in query_args is causing the
    Socrata data to not return all fields if they are blank for those
    first n rows. For now just don't use this argument.
    """
    a = time.time()
    if query_args is None:
        query_args = {}

    # get number of rows in query:
    #   default is all rows in query if limit is not specified
    if 'limit' in query_args:
        num_rows = query_args['limit']
    #   if limit is not specified get all rows
    else:
        num_rows = get_nrows_from_socrata(socrata_data_id, query_args=query_args)
    if num_rows == 0:
        print('no rows match your query')
        return

    # set chunk size based on max_chunk_area (default is 1m values):
    #   get number of columns in query
    if 'select' in query_args:
        num_cols = len(query_args['select'].split(', '))
    else:
        col_list = get_cols_socrata(socrata_data_id)
        num_cols = len(col_list)
    #   set default chunksize
    chunksize = round(max_chunk_area / num_cols)
    #   extremely wide (> 1m columns) dataframe:
    #       pull one row at a time
    if num_cols > max_chunk_area:
        chunksize = 1
    #   round chunksize to nearest 100
    if chunksize > 1000:
        chunksize = int(round(chunksize, -3))
    #   if num_rows is less than chunksize, only pull num_rows
    if chunksize > num_rows:
        chunksize = num_rows

    query_args['limit'] = chunksize

    # pull data in chunks
    client = create_socrata_client()  # modified for Lambda
    offset = 0
    num_chunks = math.ceil(num_rows / chunksize)
    print('pulling data in {} chunks of {} rows each'.format(num_chunks, chunksize))
    all_data = []
    cols = get_cols_socrata(socrata_data_id)
    for i in range(num_chunks):
        print('pulling chunk {}'.format(i))
        # content_type="csv" can throw a csv_reader field length too long error
        data = client.get(socrata_data_id, offset=offset, **query_args)
        df = pd.DataFrame(data)
        all_data.append(df)
        offset += chunksize
    client.close()
    final_df = pd.concat(all_data, ignore_index=True, sort=False)
    # set correct column types (Socrata returns everything as a string, regardless of actual column type)
    metadata = get_metadata_socrata(socrata_data_id)
    # set data types (since API returns all data as string)
    socrata_ctypes = {c['fieldName']: c['dataTypeName'] for c in metadata['columns']}
    for c in final_df:
        if socrata_ctypes.get(c, 0) == 'number':
            final_df[c] = final_df[c].astype(float)
        if not ignore_date_types:
            if socrata_ctypes.get(c, 0) == 'calendar_date':
                final_df[c] = pd.to_datetime(final_df[c])
    # if not specifying columns to return in 'select', return all cols from dataset
    # (API to json only returns columns that have at least one non-null value)
    if 'select' not in query_args:
        missing_cols = set(cols).difference(set(final_df))
        for c in missing_cols:
            final_df[c] = np.nan
        final_df = final_df[cols]
    b = time.time()
    #print('took {}'.format(print_runtime(b - a)))
    return final_df


def post_df_to_s3(df, bucket, key):
    """
    Given a pandas DataFrame, S3 bucket name, and S3 dest filename,
    posts a dataframe as a csv in S3.

    This function is from https://stackoverflow.com/a/40615630
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())
    print('dataframe on S3 at {}:{}'.format(bucket, key))
    
 
    
#############################Functions which construct use cases#######################################################
# Updated/Entered at 10/25/2019
def use_cases(df_old, sample_idx=[1341, 4833, 4059, 3755, 4114, 4000, 4148, 4499]):
    """
    Multiple use cases packed into one function to test "CheckUpdates" class
    """

    df_updated = df_old.copy()
    # Use cases for passing

    # Add rows to dataset
    duplicate_val_1 = df_updated.loc[0].copy()
    duplicate_val_1['recid'] = 'thiswasadded1-f595-4fa8-8da2-975dfae46dc4'
    duplicate_val_1['edit_date'] = pd.Timestamp('2019-10-21')
    duplicate_val_2 = df_updated.loc[0].copy()
    duplicate_val_2['recid'] = 'thiswasadded2-f595-4fa8-8da2-975dfae46dc4'
    duplicate_val_2['edit_date'] = pd.Timestamp('2019-10-21')
    df_updated = df_updated.append([duplicate_val_1, duplicate_val_2])

    # Drop rows from dataset
    df_updated = df_updated.drop([sample_idx[1], sample_idx[0]], axis=0)

    # Turn NaN values into non-nan values (2 examples)
    existing_values = df_updated.iloc[sample_idx[2]]
    existing_values['source'] = 123
    existing_values['units_per_lot'] = 12
    existing_values['minimum_lot_sqft'] = 10000
    existing_values['edit_date'] = pd.Timestamp('2019-10-25')
    df_updated.iloc[sample_idx[2]] = existing_values

    existing_val_2 = df_updated.iloc[sample_idx[3]]
    existing_val_2['source'] = 25
    existing_val_2['max_far'] = 16
    existing_val_2['edit_date'] = pd.Timestamp('2019-10-25')
    df_updated.iloc[sample_idx[3]] = existing_val_2


    # Turn values into other values (2 examples)
    existing_values3 = df_updated.iloc[sample_idx[4]]
    existing_values3['regional_lu_class'] = 123
    existing_values3['max_dua'] = 12
    existing_values3['edit_date'] = pd.Timestamp('2019-10-25')
    df_updated.iloc[sample_idx[4]] = existing_values3

    existing_val_4 = df_updated.iloc[sample_idx[5]]
    existing_val_4['regional_lu_class'] = 12
    existing_val_4['max_dua'] = 16
    existing_val_4['max_far'] = 26
    existing_val_4['edit_date'] = pd.Timestamp('2019-10-25')
    df_updated.iloc[sample_idx[5]] = existing_val_4

    #Out of bounds
    existing_values5 = df_updated.iloc[sample_idx[6]]
    existing_values5['regional_lu_class'] = 123
    existing_values5['max_dua'] = 5000
    existing_values5['building_height'] = 50
    existing_values5['edit_date'] = pd.Timestamp('2019-10-25')
    df_updated.iloc[sample_idx[6]] = existing_values5

    #Changing String Values
    existing_values6 = df_updated.iloc[sample_idx[7]]
    existing_values6['regional_lu_class'] = 123
    existing_values6['max_dua'] = 5000
    existing_values6['building_height'] = 50
    existing_values6['edit_date'] = pd.Timestamp('2019-10-25')
    existing_values6['zn_description'] = 'Testing In Place'
    df_updated.iloc[sample_idx[7]] = existing_values6

    # Use case for all fail

    return df_updated

# Updated/Entered at 11/14/2019
def use_cases_entry_2(df_old, sample_idx=[1333, 4872, 4153, 3645]):
    df_updated = df_old.copy()
    # Use cases for passing

    # Add row to dataset
    duplicate_val_1 = df_updated.loc[0].copy()
    duplicate_val_1['recid'] = 'thiswasadded1-f595-4fa8-8da2-975dfae46dc4'
    duplicate_val_1['edit_date'] = pd.Timestamp('2019-11-14')
    df_updated = df_updated.append(duplicate_val_1)
    
    #Out of bounds
    existing_values5 = df_updated.iloc[sample_idx[0]]
    existing_values5['regional_lu_class'] = 123
    existing_values5['max_dua'] = 5000
    existing_values5['building_height'] = 50
    existing_values5['edit_date'] = pd.Timestamp('2019-11-14')
    df_updated.iloc[sample_idx[0]] = existing_values5
    
    # Drop rows from dataset
    df_updated = df_updated.drop([sample_idx[1], sample_idx[0]], axis=0)
    
    # Turn values into other values (2 examples)
    existing_values3 = df_updated.iloc[sample_idx[1]]
    existing_values3['regional_lu_class'] = 123
    existing_values3['max_dua'] = 12
    existing_values3['edit_date'] = pd.Timestamp('2019-11-14')
    df_updated.iloc[sample_idx[1]] = existing_values3

    existing_val_4 = df_updated.iloc[sample_idx[2]]
    existing_val_4['regional_lu_class'] = 12
    existing_val_4['max_dua'] = 16
    existing_val_4['max_far'] = 26
    existing_val_4['edit_date'] = pd.Timestamp('2019-11-14')
    df_updated.iloc[sample_idx[2]] = existing_val_4
    
    #Changing String Values
    existing_values6 = df_updated.iloc[sample_idx[3]]
    existing_values6['regional_lu_class'] = 123
    existing_values6['max_dua'] = 5000
    existing_values6['building_height'] = 50
    existing_values6['edit_date'] = pd.Timestamp('2019-11-14')
    existing_values6['zn_description'] = 'Zone is looking good!'
    df_updated.iloc[sample_idx[3]] = existing_values6

    return df_updated

# Updated/Entered at 1/1/2020
def use_cases_entry_3(df_old, sample_idx=[123, 147, 1000]):
    df_updated = df_old.copy()
    
    # Turn values into other values (2 examples)
    existing_values3 = df_updated.iloc[sample_idx[1]]
    existing_values3['regional_lu_class'] = 123
    existing_values3['max_dua'] = 12
    existing_values3['edit_date'] = pd.Timestamp('2020-1-1')
    df_updated.iloc[sample_idx[1]] = existing_values3

    existing_val_4 = df_updated.iloc[sample_idx[2]]
    existing_val_4['regional_lu_class'] = 12
    existing_val_4['max_dua'] = 16
    existing_val_4['max_far'] = 26
    existing_val_4['edit_date'] = pd.Timestamp('2020-1-1')
    df_updated.iloc[sample_idx[2]] = existing_val_4
    
    return df_updated

def grab_update_indices(test) -> list:
    """
    One time use function that finds creates the jurisdiction update simulation from the old/new_df implementation
    originally created to test the class.
    """
    freq = [val for val in test.groupby(test.index).count()['recid']]
    indices = test.groupby(test.index).count()['recid'].index

    target_idx = []

    for f, i in zip(freq, indices):
        if f == 2:
            target_idx.append(i)
            
    return target_idx


    
if __name__ == "__main__":
    main()