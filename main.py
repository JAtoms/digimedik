import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import webbrowser
import os
import pandas as pd
from io import StringIO
import numpy as np
from dash import Dash, dcc, html
import plotly.graph_objs as go
import matplotlib.pyplot as plt

###########################################
#####        Data visualisation        ####
###########################################

def prepare_colors(df):
    # Prepare colors based on bolus trigger conditions
    df_autoXmanual = df.groupby("Time")["bolus_Trigger"].max()
    colors = replaced_array = np.where(df_autoXmanual.values == 0, 'white',
                    np.where(df_autoXmanual.values == 1, 'blue',
                    np.where(df_autoXmanual.values == 2, 'purple',
                    np.where(df_autoXmanual.values == 3, 'red', df_autoXmanual.values))))
    return colors

def create_additional_plot(df):
    # Example function to create a second plot
    fig = go.Figure()
    
    # Add traces for this additional plot (for demonstration, using basal insulin)
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['basal_Insulin Delivered (U)'],
        mode='lines',
        name='Basal Insulin',
        line=dict(color='blue')
    ))
    
    # Update layout for the additional plot
    fig.update_layout(
        title='Additional Plot Title',
        xaxis_title='Time of Day',
        yaxis_title='Basal Insulin',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_basal_quartile_plot(fig, df_basal_insulin):
    # Add basal insulin quartiles and median to the plot
    fig.add_trace(go.Scatter(
        x=df_basal_insulin['Time'],
        y=df_basal_insulin['Median'],
        mode='lines',
        name='Basal Median',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=df_basal_insulin['Time'],
        y=df_basal_insulin['Q1'],
        mode='lines',
        name='Basal 25-75%',
        line=dict(color='orange', dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_basal_insulin['Time'],
        y=df_basal_insulin['Q3'],
        mode='lines',
        line=dict(color='orange', dash='dot'),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.3)',
        name='Basal 25-75%'
    ))

def create_bolus_segments_plot(fig, df_bolus_insulin, colors):
    # Plot bolus insulin segments with markers and color coding
    for i in range(len(df_bolus_insulin) - 1):
        x_segment = [df_bolus_insulin['Time'].iloc[i], df_bolus_insulin['Time'].iloc[i + 1]]
        y_segment = [df_bolus_insulin['Max'].iloc[i], df_bolus_insulin['Max'].iloc[i + 1]]
        color = colors[i]
        
        fig.add_trace(go.Scatter(
            x=x_segment,
            y=y_segment,
            mode='lines+markers',
            yaxis="y2",
            line=dict(color=color, width=2),
            showlegend=False,
            marker=dict(symbol='circle', size=8, color=color, opacity=0.8)
        ))

    # Add a dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],  # No actual points
        y=[None],
        mode='lines',
        name='Bolus - automatic',
        line=dict(color='blue')
    ))

    # Add a dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],  # No actual points
        y=[None],
        mode='lines',
        name='Bolus - manual pre food',
        line=dict(color='purple')
    ))

    # Add a dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],  # No actual points
        y=[None],
        mode='lines',
        name='Bolus - manual no food',
        line=dict(color='red')
    ))

def create_daily_glycemia_summary_plot(df):
    fig = go.Figure()
    df_glucose = summarize_df(df, 'cgm_CGM Glucose Value (mmol/l)')

    # Add median line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['Median'],
        mode='lines',
        name='Median',
        line=dict(color='green')
    ))

    # Add first quartile (Q1) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['Q1'],
        mode='lines',
        name='25-75%',
        line=dict(color='orange', dash='dot'),
        showlegend=False  # Hide legend for this trace
    ))

    # Add third quartile (Q3) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['Q3'],
        mode='lines',
        line=dict(color='orange', dash='dot'),
        fill='tonexty',  # This fills the area between the Q1 and Q3 lines
        fillcolor='rgba(255, 165, 0, 0.3)',  # Set fill color with transparency
        name='25-75%'  # Use same label for legend
    ))

    # Add first quartile (D1) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['D1'],
        mode='lines',
        name='10-90%',
        line=dict(color='blue', dash='dot'),
        showlegend=False  # Hide legend for this trace
    ))

    # Add third quartile (D9) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['D9'],
        mode='lines',
        line=dict(color='blue', dash='dot'),
        fill='tonexty',  # This fills the area between the Q1 and Q3 lines
        fillcolor='rgba(0, 165, 255, 0.1)',  # Set fill color with transparency
        name='10-90%'  # Use same label for legend
    ))

    # Add first quartile (D1) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['Max'],
        mode='lines',
        name='Min/Max',
        line=dict(color='orange', dash='dash'),
        showlegend=False  # Hide legend for this trace,
    ))

    # Add third quartile (D9) line
    fig.add_trace(go.Scatter(
        x=df_glucose['Time'],
        y=df_glucose['Min'],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name='Min/Max'  # Use same label for legend
    ))

    # Update layout
    fig.update_layout(
        title='Glicemia Summary',
        xaxis_title='Time of Day',
        yaxis_title='Glucose [mmol/l]',
        xaxis=dict(tickmode='array'),
        legend=dict(x=0.01, y=0.99)
    )

    return fig

def create_daily_insulin_summary_plot(df):
    # Summarize data and prepare colors
    df_basal_insulin = summarize_df(df, 'basal_Insulin Delivered (U)')
    df_bolus_insulin = summarize_df(df, 'bolus_Insulin Delivered (U)')
    colors = prepare_colors(df)

    # Create the plot with basal and bolus insulin data
    fig = go.Figure()
    create_basal_quartile_plot(fig, df_basal_insulin)
    create_bolus_segments_plot(fig, df_bolus_insulin, colors)
    
    # Update layout
    fig.update_layout(
        title='Insulin Summary',
        xaxis_title='Time of Day',
        yaxis_title='Basal [U]',
        yaxis2=dict(
            title='Bolus [U]',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(tickmode='array'),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def plot_visuals(df, output_text):
    # Initialize Dash app
    global dash_app
    dash_app = Dash(__name__)

    # Define layout and plot
    dash_app.layout = html.Div([
        html.H1("Insulin pump treatment overview"),
        dcc.Graph(figure=create_daily_insulin_summary_plot(df)),
        dcc.Graph(figure=create_daily_glycemia_summary_plot(df))  # New plot added here
    ])

    # Start the HTTP server
    return dash_app

###########################################
#####        Data processing           ####
###########################################

def summarize_df(df, col):
    df_summary = df.groupby('Time').agg(
        Median=(col, 'median'),
        Q1=(col, lambda x: x.quantile(0.25)),
        Q3=(col, lambda x: x.quantile(0.75)),
        D1=(col, lambda x: x.quantile(0.1)),
        D9=(col, lambda x: x.quantile(0.9)),
        Min=(col, 'min'),
        Max=(col, 'max'),
    ).reset_index()
    return df_summary

def convert_df_to_json(df):
    json_data = df.to_json(orient="columns")
    return json_data

def process_csv_to_df(csv_dict, output_text):
    # Initialize max and min datetime variables to track the date range across all dataframes
    max_dt = pd.to_datetime("1900-01-01 12:00:00")
    min_dt = pd.to_datetime("2099-01-01 12:00:00")

    update_ui("Starting processing of CSV files...", output_text)

        # Iterate through each item in the dictionary
    for key, value in csv_dict.items():
        if isinstance(value, list):  # If the item is a list, it means data is missing for that key
            update_ui(f"Error: Missing data for required key '{key}'. Stopping processing.")
            messagebox.showerror("Processing Error", f"Missing data for '{key}'. Please check the input files and try again.")
            return None  # Stop further processing if data is missing


        update_ui(f"Processing '{key}' data...", output_text)
        
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

    update_ui("Combining data into a unified time range...", output_text)
    
    # Create a new DataFrame with a time range from the minimum to maximum timestamp across all dataframes
    df = pd.DataFrame({'DateTime': pd.date_range(start=min_dt, end=max_dt, freq='1min')})
    df = df.set_index('DateTime')
    for key, value in csv_dict.items():
        df = df.join(value)

    update_ui("Calculating basal insulin delivery...", output_text)
    # Calculate insulin delivered by dividing basal rate per minute and forward-fill missing values

    df["basal_Insulin Delivered (U)"] = df["basal_Rate"] / 60
    df["basal_Insulin Delivered (U)"] = df["basal_Insulin Delivered (U)"].ffill()

    update_ui("Setting up bolus trigger conditions...", output_text)
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

    update_ui("Filling missing bolus insulin values...", output_text)
    # Replace NaN values in "bolus_Insulin Delivered (U)" with 0
    df["bolus_Insulin Delivered (U)"] = df["bolus_Insulin Delivered (U)"].fillna(0)

    update_ui("Interpolating missing CGM glucose values...", output_text)
    # Interpolate missing values in "cgm_CGM Glucose Value (mmol/l)" column based on time
    df["cgm_CGM Glucose Value (mmol/l)"] = df["cgm_CGM Glucose Value (mmol/l)"].interpolate(method='time')

    update_ui("Identifying hypoglycemic events...", output_text)
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

    update_ui("Processing complete.", output_text)
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

    return dataframes_by_type

###########################################
#####             UI                   ####
###########################################

# Function to update the UI with messages
def update_ui(message, output_text):
    output_text.insert(tk.END, message + "\n")  # Add message to the scrolled text widget
    output_text.yview(tk.END)  # Auto-scroll to the bottom
    root.update()  # Update the root to refresh the UI immediately

# Create a complete pipeline function
def run_pipeline(folder_path, output_text):
    update_ui("Loading CSV files...", output_text)
    csv_dict = load_data(folder_path, output_text)
    
    if not csv_dict:  # Check if any data was loaded
        update_ui("No CSV files found in the selected folder.", output_text)
        return
    
    update_ui("Processing data...", output_text)
    processed_df = process_csv_to_df(csv_dict, output_text)
    
    if processed_df is None:  # Check if processing was successful
        update_ui("Data processing failed.")
        return
    
    update_ui("Visualizing data...", output_text)
    dash_app = plot_visuals(processed_df, output_text)

    update_ui("Starting Dash app...", output_text)
    # Run the Dash app in a separate thread to avoid blocking the Tkinter UI
    threading.Thread(target=dash_app.run_server, kwargs={'debug': True, 'use_reloader': False}).start()

    # Open the web browser to the Dash app
    webbrowser.open("http://localhost:8050")  # Adjust port if necessary

    update_ui("Pipeline completed successfully.", output_text)


# Function to select a folder and run the pipeline
def select_folder_and_run(output_text):
    folder_path = filedialog.askdirectory()
    if folder_path:
        update_ui(f"Selected folder: {folder_path}", output_text)
        run_pipeline(folder_path, output_text)
    else:
        messagebox.showwarning("No Selection", "No folder was selected.")


# Function to close the Tkinter window and Dash server
def on_closing():
    root.destroy() # Close the Tkinter window
    if dash_app is not None and hasattr(dash_app.server, 'shutdown'):
        # Close the Dash app server
        dash_app.server.shutdown()
        

if __name__ == '__main__':
    # Globals
    global root, keywords, server_thread
    keywords = ["alarms", "bg", "cgm", "basal", "bolus", "insulin"] # Keywords for categorizing files
    
    # Create the main window
    root = tk.Tk()
    root.title("Insulin pump treatment overview")
    root.geometry("500x400")

    # Bind the closing event to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Create a button to open the folder dialog
    select_button = tk.Button(root, text="Select folder", command=lambda: select_folder_and_run(output_text))
    select_button.pack(pady=10)

    # Create a scrolled text widget for displaying messages
    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15)
    output_text.pack(pady=10)

    # Create an Exit button
    exit_button = tk.Button(root, text="Exit application", command=on_closing)
    exit_button.pack(pady=10)

    # Run the application
    root.mainloop()