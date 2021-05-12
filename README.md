# MTC Internship Projects

From January 2021 to the present, internship partner Danh Nguyen and I have tackled multiple projects proposed by the Metropolitan Transportation Commission in their efforts to collect and standardize zoning information from jurisdictions around the Bay Area. The current overarching goal of the agency is to aggregate as much information as possible to aid in making [Plan Bay Area 2050](https://www.planbayarea.org/plan-bay-area-2050-1) (a long-term initiative addressing the economy, the environment, housing, and transportation in the Bay Area) a reality. The following projects are a combination of Research & Development as well as immediate needs of the commission.

## Project 1:  Policy Attribute Scraper Pipeline

### Goals:

1. Generate a pipeline to ease the burden of manually updating values for Zoning Atribute tables on Socrata from online documents.

2. Provide a resource which can easily generate insights into the content and structure of city zoning policy documents.

### Procedure:

1. Retrieve previously scraped policy documents from company s3 bucket.

2. Extract information by attribute using regular expression patterns.

3. Organize found attributes into a single pandas DataFrame.

4. Use document context of found attributes to identify potential zones to which they are applicable.

## Project 2: 
