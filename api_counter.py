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

# Example usage
directory_path = "groq_baselines_gemma2"  # Replace with your directory path
tries_table = count_tries(directory_path)

# Display or save the table
print(tries_table)
# Save to CSV if needed
# tries_table.to_csv("tries_summary.csv", index=False)
