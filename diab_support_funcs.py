import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import filedialog, messagebox, scrolledtext
import os
import pandas as pd
from io import StringIO

###########################################
#####        Data visualisation        ####
###########################################



###########################################
#####        Data processing           ####
###########################################

def convert_df_to_json(df):
    json_data = df.to_json(orient="columns")
    return json_data

def process_csv_to_df(csv_dict, output_text, root):
    # Function to add messages to the UI
    def update_ui(message):
        output_text.insert(tk.END, message + "\n")
        output_text.see(tk.END)
        root.update()  # Update the root to refresh the UI immediately

    # Initialize max and min datetime variables to track the date range across all dataframes
    max_dt = pd.to_datetime("1900-01-01 12:00:00")
    min_dt = pd.to_datetime("2099-01-01 12:00:00")
    
    update_ui("Starting processing of CSV files...")

        # Iterate through each item in the dictionary
    for key, value in csv_dict.items():
        if isinstance(value, list):  # If the item is a list, it means data is missing for that key
            update_ui(f"Error: Missing data for required key '{key}'. Stopping processing.")
            messagebox.showerror("Processing Error", f"Missing data for '{key}'. Please check the input files and try again.")
            return None  # Stop further processing if data is missing


        update_ui(f"Processing '{key}' data...")
        
        # Convert the "Timestamp" column to datetime format for time-based indexing
        value["Timestamp"] = pd.to_datetime(value["Timestamp"])

        # Set "Timestamp" as the index and remove the "Serial Number" column
        value = value.set_index("Timestamp").drop("Serial Number", axis=1)

        # Add a prefix to each column name to indicate its type (e.g., "alarms_data", "bg_data")
        value = value.add_prefix(f"{key}_")

        # Update max_dt and min_dt with the maximum and minimum timestamps found in this dataframe
        if max_dt < value.index.max():
            max_dt = value.index.max()
        if min_dt > value.index.min():
            min_dt = value.index.min()

        # Update the dictionary with the processed dataframe for this key
        csv_dict[key] = value

    update_ui("Combining data into a unified time range...")
    
    # Create a new DataFrame with a time range from the minimum to maximum timestamp across all dataframes
    df = pd.DataFrame({'DateTime': pd.date_range(start=min_dt, end=max_dt, freq='1min')})
    df = df.set_index('DateTime')

    update_ui("Calculating basal insulin delivery...")
    # Calculate insulin delivered by dividing basal rate per minute and forward-fill missing values
    df["basal_Insulin Delivered (U)"] = df["basal_Rate"] / 60
    df["basal_Insulin Delivered (U)"] = df["basal_Insulin Delivered (U)"].ffill()

    update_ui("Setting up bolus trigger conditions...")
    # Initialize "bolus_Trigger" column to track different bolus types
    df['bolus_Trigger'] = 0
    # Set bolus trigger based on specific conditions in 'bolus_Carbs Input (g)' and 'bolus_Carbs Ratio'
    df['bolus_Trigger'].loc[(df['bolus_Carbs Input (g)'] == 0)] = 1  # Automatic bolus
    df['bolus_Trigger'].loc[(df['bolus_Carbs Input (g)'] != 0) & 
                            (df['bolus_Carbs Input (g)'].notna()) & 
                            (df['bolus_Carbs Ratio'] != 0) & 
                            (df['bolus_Carbs Ratio'].notna())] = 2  # Manual bolus with food
    df['bolus_Trigger'].loc[(df['bolus_Carbs Input (g)'] == 0) & 
                            (df['bolus_Carbs Input (g)'].notna()) & 
                            (df['bolus_Carbs Ratio'] != 0) & 
                            (df['bolus_Carbs Ratio'].notna())] = 3  # Manual bolus without food

    update_ui("Filling missing bolus insulin values...")
    # Replace NaN values in "bolus_Insulin Delivered (U)" with 0
    df["bolus_Insulin Delivered (U)"] = df["bolus_Insulin Delivered (U)"].fillna(0)

    update_ui("Interpolating missing CGM glucose values...")
    # Interpolate missing values in "cgm_CGM Glucose Value (mmol/l)" column based on time
    df["cgm_CGM Glucose Value (mmol/l)"] = df["cgm_CGM Glucose Value (mmol/l)"].interpolate(method='time')

    update_ui("Identifying hypoglycemic events...")
    # Initialize a "HypoEvent" column to identify hypoglycemic events
    df["HypoEvent"] = 0

    # Define a time window (120 minutes) for hypoglycemia detection after a bolus event
    time_window = 120
    # Check each row in the DataFrame to identify hypoglycemic events
    for idx, row in df.reset_index().iterrows():
        if row["bolus_Trigger"] != 0:  # If there's a bolus event
            # Check if glucose levels drop below 3.9 mmol/l within the time window after a bolus
            if any(df["cgm_CGM Glucose Value (mmol/l)"][idx:idx+time_window] < 3.9):
                df["HypoEvent"][idx] = 1  # Mark "HypoEvent" if hypoglycemia occurs

    # Add a "Time" column that contains only the time (HH:MM:SS) for each index timestamp
    df["Time"] = (df.index).time

    update_ui("Processing complete.")
    return df

def process_to_csv(data):
    csv_data = data['csvData']
    df_list = []

    for key, value in csv_data.items():
        for file in value:
            csv_content = "\n".join(file.split("\r\n")[1:])
            df = pd.read_csv(StringIO(csv_content))
            df_list.append(df)
        if not(df_list):
            continue
        combined_df = pd.concat(df_list, ignore_index=True)
        csv_data[key] = combined_df
        df_list = []
    return csv_data

def load_data(folder_path, output_text):
        # Dictionary to hold lists of DataFrames by type
    dataframes_by_type = {key: [] for key in keywords}

    # Walk through folder and subfolders to find CSV files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # Check each keyword and categorize the file
                for key in keywords:
                    if key in file:  # Check if keyword is in filename
                        message = f"Loading {file_path} for type '{key}'\n"
                        output_text.insert(tk.END, message)
                        output_text.see(tk.END)
                        
                        # Read CSV, skipping the first row
                        df = pd.read_csv(file_path, skiprows=1)
                        dataframes_by_type[key].append(df)
                        break

    # Merge and save each category of DataFrame into one CSV file
    for key, dfs in dataframes_by_type.items():
        if dfs:  # Only proceed if there are DataFrames to merge
            merged_df = pd.concat(dfs, ignore_index=True)
            dataframes_by_type[key] = merged_df

            # output_file = f"{key}_merged.csv"
            # merged_df.to_csv(output_file, index=False)
            
            # message = f"Saved merged data for '{key}' to {output_file}\n"
            # output_text.insert(tk.END, message)
            # output_text.see(tk.END)
        else:
            message = f"No files found for type '{key}'\n"
            output_text.insert(tk.END, message)
            output_text.see(tk.END)

    process_csv_to_df(dataframes_by_type)