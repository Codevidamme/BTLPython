import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#-------------------------------------------------------------------------------
# CONFIGURATION - USER: PLEASE VERIFY AND EDIT THESE BASED ON YOUR results.csv
#-------------------------------------------------------------------------------

# List of ALL STATISTIC COLUMNS from results.csv that need to be analyzed.
# These names MUST MATCH the column headers in your 'results.csv' file.
# This list is based on the statistics mentioned in the PDF for Bài 1 & Bài 2.
ALL_STAT_COLUMNS_TO_ANALYZE = [
    'Age', 'MP', 'Starts', 'Min',  # Player Info & Playing Time # MP is Matches Played, Min is Minutes
    'Gls', 'Ast', 'CrdY', 'CrdR',  # Performance # Goals, Assists, Yellow Cards, Red Cards
    'xG', 'xAG',                  # Expected Stats # Expected Goals, Expected Assisted Goals
    'PrgC', 'PrgP', 'PrgR',       # Progression # Progressive Carries, Progressive Passes, Progressive Passes Received
    # Per 90 stats: Names depend on how Part I scraper saved them.
    # Common FBRef: 'Gls.1', 'Ast.1', 'xG.1', 'xAG.1' for Gls, Ast, xG, xAG per 90.
    # PDF asks for: Gls, Ast, xG, XGA per 90. Assuming XGA means xAG.
    # Based on user's results.csv: 'Per90_Gls', 'Per90_Ast', 'Per90_xG90', 'Per90_xAG' might be actual names
    # For now, using the placeholder names from the original script. User needs to verify.
    'Gls.1', 'Ast.1', 'xG.1', 'xAG.1', # Placeholder, check actual names like 'Per90_Gls'
    # Goalkeeping Stats:
    'GA90',       # Goals Against per 90
    'Save%',      # Save Percentage
    'CS%',        # Clean Sheet Percentage
    'PKsv%',      # Penalty Kick Save Percentage (FBRef might use Save%.1 or Penalty_Save%)
    # Shooting Stats:
    'SoT%',       # Shots on Target %
    'SoT/90',     # Shots on Target per 90
    'G/Sh',       # Goals per Shot
    'Dist',       # Average Shot Distance
    # Passing Stats:
    'Cmp',        # Passes Completed (Total)
    'Cmp%',       # Pass Completion % (Total)
    'PrgDist',    # Progressive Passing Distance (Total) (PDF also mentions TotDist, user's CSV has 'Total_Cmp%' and 'TotDist' for passes)
    # Pass Completion % for Short, Medium, Long.
    # User's CSV has 'Cmp_Short%', 'Cmp_Median%', 'Cmp_Long%'
    'Cmp_Short%', 'Cmp_Median%', 'Cmp_Long%', # Adjusted to likely names from user's CSV
    # Expected Passing Stats from PDF (KP, 1/3, PPA, CrsPA, PrgP)
    'KP', '1/3_Expected', 'PPA', 'CrsPA', 'PrgP_Expected', # '1/3_Expected' and 'PrgP_Expected' might be different in CSV
    # Goal and Shot Creation
    'SCA', 'SCA90', 'GCA', 'GCA90',
    # Defensive Actions
    'Tkl', 'TklW', 'Att_C', 'Lost_C', 'Blocks', 'Sh', 'Pass', 'Int', # 'Att_C', 'Lost_C' for challenges, Sh,Pass for blocks
    # Possession
    'Touches', 'DefPen_', 'Def3rd', 'Mid_3rd', 'Att_3rd', 'Att_pen', # DefPen_ from user's CSV
    'Take-Ons_Att', 'Succ%', 'Tkld%', # Succ% and Tkld% for Take-ons
    'Carries', 'ProDist', 'PrgC_cr', '1/3_Cr', 'CPA', 'Mis', 'Dis', # PrgC_cr, 1/3_Cr from user's CSV
    # Receiving
    'Rec', 'PrgR_R', # PrgR_R from user's CSV
    # Miscellaneous
    'Fls', 'Fld', 'Off', 'Crs', 'Recov', # Fouls, Fouled, Offsides, Crosses, Recoveries
    'Won', 'Lost', 'Won%' # Aerial duels Won, Lost, Won%
]

# Column name for Player Name in results.csv
PLAYER_NAME_COLUMN = 'Name' # From user's results.csv

# Column name for Team/Squad in results.csv
TEAM_COLUMN = 'Team'  # From user's results.csv (was 'Squad' in original template)

# Files and Directories
INPUT_CSV_PATH = 'results.csv'
OUTPUT_CSV_PATH = 'results2.csv'
HISTOGRAMS_OUTPUT_DIR = 'histograms_output'
TOP_3_OUTPUT_TXT_PATH = 'top_3.txt' # New output file for top/bottom 3 players

# For team performance analysis: stats where lower is better
LOWER_IS_BETTER_STATS = ['CrdY', 'CrdR', 'GA90', 'Fls', 'Lost_C', 'Dis', 'Lost'] # Adjusted 'Err', 'OG' if not present, Added Lost_C, Dis, Lost

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def load_and_prepare_data(csv_filepath, stat_definition_list, team_col_primary, player_name_col):
    """Loads data, handles 'N/a', 'NaN' strings, commas in numbers, and converts to numeric."""
    global TEAM_COLUMN # Allow modification of global TEAM_COLUMN if primary not found
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return None, []
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return None, []

    # Ensure player name column exists
    if player_name_col not in df.columns:
        print(f"Error: Player name column '{player_name_col}' not found in the CSV.")
        return None, []

    # Handle 'N/a' strings, and also 'NaN' strings if any, convert to actual np.nan
    for col in df.columns:
        if df[col].dtype == 'object':
            df.loc[:, col] = df[col].replace({'N/a': np.nan, 'NaN': np.nan, 'nan': np.nan}, regex=False)
            # Attempt to remove commas from numbers if column is still object type
            # and is in our list of columns to analyze (or is 'Min' which needs it)
            if col in stat_definition_list or col == 'Min': # 'Min' from PDF, or 'Minutes' from user CSV
                try:
                    # Only apply to columns that are still objects and might contain numbers with commas
                    if df[col].dtype == 'object':
                         df.loc[:, col] = df[col].astype(str).str.replace(',', '', regex=False)
                except Exception as e:
                    print(f"Note: Could not remove commas from column {col}, or it's not string type. Error: {e}")


    # Identify actual statistic columns present in the DataFrame
    actual_stat_cols_in_df = [col for col in stat_definition_list if col in df.columns]
    
    missing_cols = [col for col in stat_definition_list if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: The following defined statistic columns were NOT found in '{csv_filepath}':")
        for col in missing_cols:
            print(f"  - {col}")
        print("These columns will be skipped. Please check 'ALL_STAT_COLUMNS_TO_ANALYZE'.\n")

    if not actual_stat_cols_in_df:
        print(f"Critical Error: No defined statistic columns (from ALL_STAT_COLUMNS_TO_ANALYZE) were found in {csv_filepath}.")
        print("Please ensure 'ALL_STAT_COLUMNS_TO_ANALYZE' in the script matches your CSV column names.")
        return None, []

    # Convert identified statistic columns to numeric
    for col in actual_stat_cols_in_df:
        if df[col].dtype == 'object': # If still object after 'N/a' and comma replacement
            if df[col].str.contains('%', na=False).any():
                df.loc[:, col] = df[col].str.rstrip('%').astype('float', errors='ignore') / 100.0
            else:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        elif pd.api.types.is_numeric_dtype(df[col]):
            pass # No conversion needed
        else: # Other types, try to coerce
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    # Specifically handle 'Min' column if it's named differently in ALL_STAT_COLUMNS_TO_ANALYZE
    # but is 'Minutes' in the CSV (as per user's CSV structure)
    minutes_col_in_csv = 'Minutes' # From user's results.csv
    if minutes_col_in_csv in df.columns and minutes_col_in_csv not in actual_stat_cols_in_df:
        if df[minutes_col_in_csv].dtype == 'object':
             df.loc[:, minutes_col_in_csv] = df[minutes_col_in_csv].astype(str).str.replace(',', '', regex=False)
        df.loc[:, minutes_col_in_csv] = pd.to_numeric(df[minutes_col_in_csv], errors='coerce')
        if minutes_col_in_csv not in ALL_STAT_COLUMNS_TO_ANALYZE: # If 'Minutes' was not in the list but 'Min' was
             if 'Min' in ALL_STAT_COLUMNS_TO_ANALYZE and 'Min' not in df.columns:
                 df.rename(columns={minutes_col_in_csv: 'Min'}, inplace=True)
                 if 'Min' not in actual_stat_cols_in_df : actual_stat_cols_in_df.append('Min')
                 print(f"Processed '{minutes_col_in_csv}' as 'Min'.")


    print(f"Successfully processed. Identified numeric columns for analysis: {actual_stat_cols_in_df}")

    # Check for team column
    if team_col_primary not in df.columns:
        print(f"Warning: Primary team column '{team_col_primary}' not found.")
        # Fallback logic already present in the original script, which is fine.
        # The global TEAM_COLUMN will be updated if 'Team' or 'Squadra' is found.
        if 'Team' in df.columns:
            TEAM_COLUMN = 'Team' 
            print(f"Using fallback 'Team' as the team column.")
        elif 'Squad' in df.columns: # Common FBRef name
            TEAM_COLUMN = 'Squad'
            print(f"Using fallback 'Squad' as the team column.")
        elif 'Squadra' in df.columns: 
            TEAM_COLUMN = 'Squadra'
            print(f"Using fallback 'Squadra' as the team column.")
        else:
            print(f"Error: Neither '{team_col_primary}' nor other fallbacks found. Please specify correct TEAM_COLUMN.")
            return None, []
    else:
        TEAM_COLUMN = team_col_primary

    return df, actual_stat_cols_in_df


def calculate_descriptive_stats(df, stat_cols_to_analyze, team_col_name):
    """Calculates median, mean, std for all players and per team."""
    results_list = []

    # Stats for ALL players
    all_stats_agg = df[stat_cols_to_analyze].agg(['median', 'mean', 'std'])
    
    all_row = {'Group': 'all'}
    for stat in stat_cols_to_analyze:
        all_row[f'Median of {stat}'] = all_stats_agg.loc['median', stat]
        all_row[f'Mean of {stat}'] = all_stats_agg.loc['mean', stat]
        all_row[f'Std of {stat}'] = all_stats_agg.loc['std', stat]
    results_list.append(all_row)

    # Stats for EACH team
    if team_col_name not in df.columns:
        print(f"Error: Team column '{team_col_name}' not found for per-team stats.")
        return pd.DataFrame(results_list) 

    teams = df[team_col_name].dropna().unique()
    teams = sorted([str(team) for team in teams]) 

    for team_name_val in teams:
        team_df = df[df[team_col_name] == team_name_val]
        if team_df.empty:
            continue
        
        team_stats_agg = team_df[stat_cols_to_analyze].agg(['median', 'mean', 'std'])
        team_row = {'Group': team_name_val}
        for stat in stat_cols_to_analyze:
            team_row[f'Median of {stat}'] = team_stats_agg.loc['median', stat]
            team_row[f'Mean of {stat}'] = team_stats_agg.loc['mean', stat]
            team_row[f'Std of {stat}'] = team_stats_agg.loc['std', stat]
        results_list.append(team_row)

    results_df = pd.DataFrame(results_list)
    
    output_cols_ordered = ['Group']
    for stat in stat_cols_to_analyze:
        output_cols_ordered.extend([f'Median of {stat}', f'Mean of {stat}', f'Std of {stat}'])
    
    for col in output_cols_ordered:
        if col not in results_df.columns:
            results_df[col] = np.nan
            
    results_df = results_df[output_cols_ordered]
    results_df.insert(0, '', range(len(results_df)))
    
    return results_df


def plot_histograms(df, stat_cols_to_analyze, team_col_name, output_dir):
    """Plots and saves histograms for each statistic."""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    for stat in stat_cols_to_analyze:
        if stat not in df.columns:
            print(f"Skipping histogram for '{stat}' as it's not in the DataFrame.")
            continue
            
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.figure(figsize=(10, 6))
        data_all_players = df[stat].dropna()
        if data_all_players.empty or data_all_players.nunique() < 2 : 
            plt.close()
        else:
            plt.hist(data_all_players, bins='auto', alpha=0.75, edgecolor='black', label='All Players', density=True)
            plt.title(f'Distribution of {stat} - All Players', fontsize=15)
            plt.xlabel(stat, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.tight_layout()
            try:
                plt.savefig(os.path.join(output_dir, f'{stat}_all_players_hist.png'))
            except Exception as e:
                print(f"Error saving histogram for {stat} (all players): {e}")
            plt.close()

        if team_col_name not in df.columns:
            continue 

        teams = df[team_col_name].dropna().unique()
        teams = sorted([str(team) for team in teams])

        for team_name_val in teams:
            plt.figure(figsize=(8, 5))
            team_data = df[df[team_col_name] == team_name_val][stat].dropna()
            
            if team_data.empty or team_data.nunique() < 2:
                plt.close()
                continue
            
            safe_team_name = "".join(c if c.isalnum() else "_" for c in str(team_name_val))

            plt.hist(team_data, bins='auto', alpha=0.75, edgecolor='black', density=True)
            plt.title(f'Distribution of {stat} - Team: {team_name_val}', fontsize=14)
            plt.xlabel(stat, fontsize=11)
            plt.ylabel('Density', fontsize=11)
            plt.tight_layout()
            try:
                plt.savefig(os.path.join(output_dir, f'{stat}_team_{safe_team_name}_hist.png'))
            except Exception as e:
                print(f"Error saving histogram for {stat} (team {safe_team_name}): {e}")
            plt.close()
            
    print(f"\nHistograms saved to '{output_dir}' directory (if data was plottable).")


def analyze_team_performance(results_summary_df, stat_cols_to_analyze, low_is_good_stats):
    """Identifies top teams per statistic and provides overall commentary."""
    print("\n--- Team Performance Analysis ---")
    
    team_results = results_summary_df[results_summary_df['Group'] != 'all'].copy()
    if team_results.empty:
        print("No team data available for performance analysis.")
        return

    print("\nTeams with highest/best mean scores for each statistic:")
    top_teams_by_stat = {}

    for stat in stat_cols_to_analyze:
        mean_col = f'Mean of {stat}'
        if mean_col not in team_results.columns:
            continue

        valid_means_df = team_results[['Group', mean_col]].dropna(subset=[mean_col])
        if valid_means_df.empty:
            top_teams_by_stat[stat] = "N/A (all means are NaN)"
            continue

        if stat in low_is_good_stats:
            best_team_idx = valid_means_df[mean_col].idxmin()
            best_team_name = valid_means_df.loc[best_team_idx, 'Group']
            best_value = valid_means_df[mean_col].min()
            print(f"- Lowest (better) {stat}: {best_team_name} ({best_value:.2f})")
            top_teams_by_stat[stat] = f"{best_team_name} ({best_value:.2f}, lower is better)"
        else:
            best_team_idx = valid_means_df[mean_col].idxmax()
            best_team_name = valid_means_df.loc[best_team_idx, 'Group']
            best_value = valid_means_df[mean_col].max()
            print(f"- Highest {stat}: {best_team_name} ({best_value:.2f})")
            top_teams_by_stat[stat] = f"{best_team_name} ({best_value:.2f})"

    print("\n--- Overall Best Performing Team (Qualitative Assessment) ---")
    team_scores = {}
    for stat_name, result_text in top_teams_by_stat.items():
        if "N/A" in result_text:
            continue
        
        current_team_name = result_text.split(" (")[0]
        is_positive_stat = stat_name not in low_is_good_stats
        is_best_in_low_good_stat = stat_name in low_is_good_stats and "lower is better" in result_text

        if is_positive_stat or is_best_in_low_good_stat:
            team_scores[current_team_name] = team_scores.get(current_team_name, 0) + 1
            
    if not team_scores:
        print("Insufficient data or rankings to suggest an overall best team.")
    else:
        sorted_teams = sorted(team_scores.items(), key=lambda item: item[1], reverse=True)
        print("Based on a simple count of leading positions in key statistics:")
        for i, (team, score) in enumerate(sorted_teams):
            print(f"{i+1}. {team}: Ranked best in {score} analyzed categories.")

        if sorted_teams:
            best_overall_team_candidate = sorted_teams[0][0]
            print(f"\nBased on this simplified metric, '{best_overall_team_candidate}' shows strong performance across multiple statistics.")
        else:
            print("\nCould not determine a single leading team with this simple metric.")

    print("\nYour analysis should consider the varying importance of statistics.")
    print("For instance, high 'Gls.1' (Goals per 90) and 'xG.1' (Expected Goals per 90) ")
    print("combined with low 'GA90' (Goals Against per 90) and low 'CrdR' (Red Cards) ")
    print("would strongly indicate a top-performing team.")
    print("This part requires your critical thinking and justification based on the data.")

#-------------------------------------------------------------------------------
# NEW FUNCTION: Calculate and Save Top/Bottom 3 Players per Statistic
#-------------------------------------------------------------------------------
def calculate_and_save_top_bottom_players(df, stat_cols, player_name_col, team_col, output_filepath):
    """
    Identifies the top 3 players with the highest and lowest scores for each statistic
    and saves the result to a text file.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Skipping top/bottom player analysis.")
        return
    if not stat_cols:
        print("No statistic columns provided. Skipping top/bottom player analysis.")
        return
    if player_name_col not in df.columns:
        print(f"Player name column '{player_name_col}' not found. Skipping top/bottom player analysis.")
        return
    if team_col not in df.columns:
        print(f"Team column '{team_col}' not found. Skipping top/bottom player analysis.")
        return

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("Top 3 Players (Highest and Lowest Scores) per Statistic\n")
            f.write("=========================================================\n\n")

            for stat in stat_cols:
                if stat not in df.columns:
                    print(f"Statistic column '{stat}' not found in DataFrame. Skipping for top/bottom analysis.")
                    continue

                f.write(f"Statistic: {stat}\n")
                f.write("-" * (len(stat) + 12) + "\n")

                # Create a temporary DataFrame with relevant columns and drop NaNs for the current stat
                temp_df = df[[player_name_col, team_col, stat]].copy()
                temp_df.dropna(subset=[stat], inplace=True)

                if temp_df.empty:
                    f.write("  No data available for this statistic after removing NaNs.\n\n")
                    continue

                # Top 3 Highest
                top_3_highest = temp_df.sort_values(by=stat, ascending=False).head(3)
                f.write("  Top 3 (Highest Scores):\n")
                if not top_3_highest.empty:
                    for index, row in top_3_highest.iterrows():
                        f.write(f"    - {row[player_name_col]} ({row[team_col]}): {row[stat]:.2f}\n")
                else:
                    f.write("    - Not enough data for top 3 highest.\n")

                # Top 3 Lowest
                top_3_lowest = temp_df.sort_values(by=stat, ascending=True).head(3)
                f.write("  Top 3 (Lowest Scores):\n")
                if not top_3_lowest.empty:
                    for index, row in top_3_lowest.iterrows():
                        f.write(f"    - {row[player_name_col]} ({row[team_col]}): {row[stat]:.2f}\n")
                else:
                    f.write("    - Not enough data for top 3 lowest.\n")
                
                f.write("\n") # Add a newline for readability between stats

        print(f"\nTop/bottom 3 players per statistic saved to '{output_filepath}'")

    except Exception as e:
        print(f"Error writing top/bottom 3 players to file '{output_filepath}': {e}")


#-------------------------------------------------------------------------------
# MAIN EXECUTION
#-------------------------------------------------------------------------------
def main():
    print(f"Starting data processing for 'Bài 2' using '{INPUT_CSV_PATH}'.")
    print(f"IMPORTANT: Ensure 'ALL_STAT_COLUMNS_TO_ANALYZE' and 'TEAM_COLUMN' match your CSV structure.")

    # Note: TEAM_COLUMN is used as the primary, PLAYER_NAME_COLUMN is also passed
    df_cleaned, actual_cols = load_and_prepare_data(INPUT_CSV_PATH, ALL_STAT_COLUMNS_TO_ANALYZE, TEAM_COLUMN, PLAYER_NAME_COLUMN)

    if df_cleaned is None or not actual_cols:
        print("\nExiting due to errors in data loading or no valid statistic columns found.")
        return

    print(f"\nUsing team column: '{TEAM_COLUMN}' for per-team analysis.")
    print(f"Analyzing the following statistics: {actual_cols}")

    # --- NEW: Calculate and save top/bottom 3 players ---
    # Ensure PLAYER_NAME_COLUMN is correctly identified or passed if it can vary
    # For this script, it's defined as a global constant based on user's results.csv
    calculate_and_save_top_bottom_players(df_cleaned, actual_cols, PLAYER_NAME_COLUMN, TEAM_COLUMN, TOP_3_OUTPUT_TXT_PATH)
    # --- END NEW ---

    # Calculate descriptive statistics
    summary_stats_df = calculate_descriptive_stats(df_cleaned, actual_cols, TEAM_COLUMN)
    
    # Save results to results2.csv
    if not summary_stats_df.empty:
        try:
            summary_stats_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
            print(f"\nDescriptive statistics saved to '{OUTPUT_CSV_PATH}'")
        except Exception as e:
            print(f"Error saving '{OUTPUT_CSV_PATH}': {e}")
    else:
        print("\nNo summary statistics were generated to save.")

    # Plot histograms
    plot_histograms(df_cleaned, actual_cols, TEAM_COLUMN, HISTOGRAMS_OUTPUT_DIR)
    
    # Identify top teams and provide basis for commentary
    if not summary_stats_df.empty:
        analyze_team_performance(summary_stats_df, actual_cols, LOWER_IS_BETTER_STATS)
    else:
        print("\nSkipping team performance analysis as no summary statistics were generated.")
    
    print("\nProcessing complete for 'Bài 2'.")

if __name__ == '__main__':
    main()
