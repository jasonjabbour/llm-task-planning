import os
import pandas as pd

def count_tries(directory):
    """
    Counts the number of tries for each level based on the directory structure.

    Parameters:
        directory (str): Path to the main directory containing level subdirectories.

    Returns:
        pd.DataFrame: A table summarizing the number of tries for each level.
    """
    results = []

    for level_dir in sorted(os.listdir(directory)):
        level_path = os.path.join(directory, level_dir)

        if os.path.isdir(level_path) and level_dir.lower().startswith("level_"):
            # Extract level number
            try:
                level_number = int(level_dir.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Count folders
            initial_count = 1 if os.path.exists(os.path.join(level_path, "initial")) else 0
            refinement_count = len([
                name
                for name in os.listdir(level_path)
                if name.startswith("lost_refinement_") and os.path.isdir(os.path.join(level_path, name))
            ])

            # Total tries
            total_tries = initial_count + refinement_count

            results.append({"Level": level_number, "Tries": total_tries})

    if not results:
        print("No valid level data found. Check the directory structure.")
        return pd.DataFrame()  # Return an empty DataFrame if no results

    # Sort and return as a DataFrame
    results_df = pd.DataFrame(results).sort_values(by="Level").reset_index(drop=True)
    return results_df

def aggregate_tries(directories):
    """
    Aggregates the number of tries for multiple directories into a single DataFrame.

    Parameters:
        directories (list of str): List of directory paths to process.

    Returns:
        pd.DataFrame: A combined table summarizing the number of tries for each level across directories.
    """
    combined_df = pd.DataFrame()

    for directory in directories:
        dir_name = os.path.basename(directory)
        tries_df = count_tries(directory)
        tries_df.rename(columns={"Tries": dir_name}, inplace=True)

        if combined_df.empty:
            combined_df = tries_df
        else:
            combined_df = pd.merge(combined_df, tries_df, on="Level", how="outer")

    combined_df.fillna(0, inplace=True)  # Replace NaN with 0 for levels not present in some directories

    # Compute the total row
    total_row = {"Level": "Total"}
    for column in combined_df.columns[1:]:
        total_row[column] = combined_df[column].sum()

    combined_df = pd.concat([combined_df, pd.DataFrame([total_row])], ignore_index=True)
    return combined_df

# Example usage
directories = ["llama33_70b_versatile", "groq_baselines_gemma1", "groq_baselines_gemma2", "mixtral-7b", "o1_preview"]  # Replace with your directory paths
combined_table = aggregate_tries(directories)

# Display or save the table
print(combined_table)
# Save to CSV if needed
combined_table.to_csv("combined_tries_summary.csv", index=False)
