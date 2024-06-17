import csv
import pandas as pd
import numpy as np

# Data loading
def read_csv(pathway):
    with open(pathway, "r", newline = "") as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    return rows

# Dataset 1: Offers and responses by course offering
def process_offers_responses(rows):
    index_table1 = None
    index_table2 = None
    
    for i, row in enumerate(rows):
        if "ref_num" in row:
            index_table1 = i
        if "offer_cc1" in row:
            index_table2 = i
            break
    
    df = rows[index_table1:index_table2]
    df = [row for row in df if any(cell.strip() for cell in row)]
    df = pd.DataFrame(df[1:], columns=df[0])
    
    df2 = rows[index_table2:]
    df2 = [row for row in df2 if any(cell.strip() for cell in row)]
    df2 = pd.DataFrame(df2[1:], columns=df2[0])
    
    return df, df2

# Dataset 2: Preference by Course Offering
def process_preference(rows):
    index_table = None
    
    for i, row in enumerate(rows):
        if "ref_num" in row:
            index_table = i
            break
    
    df3 = rows[index_table:]
    df3 = pd.DataFrame(df3[1:], columns=df3[0])
    
    return df3

# Dataset 3: SATAC Code to Faculty
def read_satac_code():
    satac_code = pd.read_csv("SATAC Programs to FoE.csv", encoding="utf-16", sep="\t")
    return satac_code

# Data cleaning
def clean_preference(preference, satac_code):
    preference["fac_code"] = preference["fac_code"].replace({"UAABLE": "ABLE", "UAHLT": "HLT", "UASET": "SET"})
    preference["country"] = preference["country"].replace({"D627E8EF-1901-FBD8-C02B-E2656CD16362": "Australia"})
    
    preference.rename(columns={"fac_code": "faculty", "abtsi": "indigenous_status"}, inplace=True)
    
    foe_mapping = dict(zip(satac_code["SATAC Program Code"], satac_code["Broad FOE"]))
    preference["broad_foe"] = preference["course_code"].map(foe_mapping)
    
    preference = preference.dropna(subset=["preference_number1"])
    
    unnecessary_columns = [
        "filing_number", "title", "surname", "givens", "middle",
        "address_type1", "address_1", "address_2", "address_3", "address_4",
        "overseas_state", "Home_phone", "Work_phone", "Mobile_phone", "Textbox83",
        "email_address", "preference_number", "stream_code", "stream_name",
        "institution_course_code", "preference_eligibility_value", "course_elig_reason",
        "subquota_selection_rank_name", "ssr_rank_order", "ssr_rank_value",
        "campus_code", "campus_short_name", "fac_name", "sch_code", "sch_name",
        "institution_short_name", "LatestY12_organisation_name", "pref_date", "postcode"
    ]
    
    preference = preference.drop(columns=unnecessary_columns)
    
    preference_details = preference.loc[:, "preference_number1":"faculty"]
    for col in preference_details.iloc[:, ::-1]:
        preference.insert(preference.columns.get_loc("y12_count"), col, preference.pop(col))
    
    preference_details = pd.concat([preference["ref_num"], preference.loc[:, "preference_number1":"broad_foe"]], axis=1)
    
    for i in range(1, 7):
        preference[f"pf_course_code_{i}"] = np.nan
        preference[f"course_short_name_{i}"] = np.nan
        preference[f"course_level_name_{i}"] = np.nan
        preference[f"offering_code_{i}"] = np.nan
        preference[f"faculty_{i}"] = np.nan
        preference[f"broad_foe_{i}"] = np.nan
        
    for i in range(1, 7):
        preference[f"pf_course_code_{i}"] = preference[f"pf_course_code_{i}"].astype(str)
        preference[f"course_short_name_{i}"] = preference[f"course_short_name_{i}"].astype(str)
        preference[f"course_level_name_{i}"] = preference[f"course_level_name_{i}"].astype(str)
        preference[f"offering_code_{i}"] = preference[f"offering_code_{i}"].astype(str)
        preference[f"faculty_{i}"] = preference[f"faculty_{i}"].astype(str)
        preference[f"broad_foe_{i}"] = preference[f"broad_foe_{i}"].astype(str)
    
    preference_group = preference.loc[:, "preference_number1":"broad_foe"]
    preference = preference.drop(columns=preference_group)
    
    preference.drop_duplicates(subset=["ref_num"], keep="first", inplace=True)
    
    for index, row in preference_details.iterrows():
        ref_num = row["ref_num"]
        preference_number = int(row["preference_number1"])
        
        preference_to_map = preference[preference["ref_num"] == ref_num]
        
        for i in range(1, 7):
            if i == preference_number:
                preference.loc[preference_to_map.index, f"pf_course_code_{i}"] = row["course_code"]
                preference.loc[preference_to_map.index, f"course_short_name_{i}"] = row["course_short_name"]
                preference.loc[preference_to_map.index, f"course_level_name_{i}"] = row["course_level_name"]
                preference.loc[preference_to_map.index, f"offering_code_{i}"] = row["offering_code"]
                preference.loc[preference_to_map.index, f"faculty_{i}"] = row["faculty"]
                preference.loc[preference_to_map.index, f"broad_foe_{i}"] = row["broad_foe"]
    
    return preference

def clean_response(response, satac_code):
    response["faculty_code"] = response["faculty_code"].replace({"UAABLE": "ABLE", "UAHLT": "HLT", "UASET": "SET"})
    response.rename(columns={"faculty_code": "offer_faculty", "round_number": "offer_round_number"}, inplace=True)
    
    foe_mapping_offer = dict(zip(satac_code["SATAC Program Code"], satac_code["Broad FOE"]))
    response["offer_broad_foe"] = response["offer_cc"].map(foe_mapping_offer)
    
    response_details = pd.concat([response["ref_num"], 
                                  response.loc[:, "offer_cc":"offer_title"],
                                  response.loc[:, "offer_sem":"response_date"],
                                  response.loc[:, "offer_boa":"offer_faculty"],
                                  response.loc[:, "offer_broad_foe"]], axis=1)
    
    return response_details

def merge_datasets(preference, response_details):
    merged_df = pd.merge(preference, response_details, on="ref_num", how="left")
    merged_df["response"] = merged_df.pop("response")
    merged_df.replace("nan", "", inplace=True)
    
    return merged_df

# Export the cleaned datasets
def export_to_csv(merged_df, response_summary):
    merged_df.to_csv("final.csv", index=False)
    response_summary.to_csv("response_summary.csv", index=False)

# Main function
def main():
    # Dataset 1: Offers and responses by course offering
    pathway1 = "Offers and responses by course offering plus offer round and offer currency.csv"
    rows1 = read_csv(pathway1)
    response, response_summary = process_offers_responses(rows1)
    response.replace("", np.nan, inplace=True)
    
    # Dataset 2: Preference by Course Offering
    pathway2 = "Preference by Course Offering.csv"
    rows2 = read_csv(pathway2)
    preference = process_preference(rows2)
    preference.replace("", np.nan, inplace=True)
    
    # Dataset 3: SATAC Code to Faculty
    satac_code = read_satac_code()
    
    # Clean preference dataset
    preference_cleaned = clean_preference(preference, satac_code)
    
    # Clean response dataset
    response_details_cleaned = clean_response(response, satac_code)
    
    # Merge datasets
    merged_df = merge_datasets(preference_cleaned, response_details_cleaned)
    
    # Export datasets
    export_to_csv(merged_df, response_summary)

if __name__ == "__main__":
    main()






































