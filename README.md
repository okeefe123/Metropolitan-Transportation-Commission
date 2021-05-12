# MTC Internship Projects

From January 2021 to the present, internship partner Danh Nguyen and I have tackled multiple projects proposed by the Metropolitan Transportation Commission in their efforts to collect and standardize zoning information from jurisdictions around the Bay Area. The current overarching goal of the agency is to aggregate as much information as possible to aid in making [Plan Bay Area 2050](https://www.planbayarea.org/plan-bay-area-2050-1) (a long-term initiative addressing the economy, the environment, housing, and transportation in the Bay Area) a reality. The following projects are a combination of Research & Development as well as solutions to the immediate needs of the commission.

# Project 1:  Policy Attribute Scraper Pipeline

### Goals:

1. Generate a pipeline to ease the burden of manually updating values for Zoning Atribute tables on Socrata from online documents.

2. Provide a resource which can easily generate insights into the content and structure of city zoning policy documents.

### Procedure:

1. Retrieve previously scraped policy documents from company s3 bucket.

2. Extract information by attribute using regular expression patterns.

3. Organize found attributes into a single pandas DataFrame.

4. Use document context of found attributes to identify potential zones to which they are applicable.

# Project 2: Attribute Classifier

### Goals:

- Create a classification model that, when given a raw document, finds the probability that a given line holds relevance to a land use attribute.

- Provide MTC with yet another metric-drive analytical tool to determine the standardization in the structure of policy documents from any jurisdiction.

### Procedure:

1. Preprocessing:
    - Transform raw text version of policy document into a table
    - Each row represents a line of the document.
    - Initial features - city, line of the policy


2. Feature Selection:
    - Tokenize each row using spaCy and lemmatize all tokens
    - Extract character count, character count, and average word length as numerical features
    - Encode cities via mean frequency
    - Vectorize the lemmatized tokens such that the model can interpret them during training/testing
    - Encode labels in the training/test set
    
### Production Model:

- Provides the probability of any given line in a document belonging to each of the assumed land use classes
- Necessary Information Format:
    1. Raw text version of policy zoning document
    2. city_frequency.json - information of the count frequency of documents used during training stage
    3. decode_labels.json - connection between integer classes and their corresponding land use attribute
  
 
# Project 3: QA/QC Incoming Jurisdiction Information:

###Goals:

    1. Given entries from a jurisdiction, track and analyze the changes from any previous table version.
        a. Determine Metrics that can be used as a reflection of valid entries
        b. Generalize to compare any updated entry with any master table enty which shares same unique key
        
    2. Find a method to track all rows potentially affected in redshift database via primary key
    
    3. Implement analysis as an AWS Lambda Function for external use.
        a. Gain a better understanding of "Layers" to bootstrap libraries in a space efficient manner
        b. Implement call capabilities both from a Python Environment and through a REST API
        
    4. Store warning and update logs into appropriate socrata tables for future reference.


### Assumptions:
    1. The tables ingested by this pipeline have been pulled from Socrata (and thus that values for all features are consistent in datatype)

    2. Jurisdiction updates are presented to the function as a json file.
    
    3. Aforementioned presentation shares same schema as old table
        a. Full table with updated columns
        b. Updated columns themselves
                
    4. If a row was added in the update, it was assigned a unique primary key
    
    5. A row's recid cannot be changed (since it's used as the primary key between old and new dataframe)
    
    6. Columns of type "float" are continuous (not ordinal/nominal categories)
    
    7. The fifth percentile to the 95th percentile of master socrata dataframe numerical columns is sufficient metric for "normal" updates
    
    8. Novel string updates when compared to old column counterparts are suspicious enough to raise warnings
    

### Output:

    1. Update table log with schema:
        - recid, city_name, cols_updated, updated_from_NaN, changed_vals, out_of_range, warnings, editor, edit_timestamp, edit_type (C/U/D), main_version, old_vals, new_vals
        
    2. Warning log with schema:
        - city, editor, recid, warn
