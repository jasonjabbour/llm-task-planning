#!/bin/bash

# Output file for the JSON results
output_file="plans_demo_LEVELS.json"

# Initialize the JSON file
echo "{" > "$output_file"

# Path to Fast Downward
FAST_DOWNWARD="python /home/z/downward/fast-downward.py"

# Loop over levels
for level in {0..13}; do
    echo "Processing level $level..."

    # Define the problem file and the plan file
    problem_file="pddl_problem_files/level${level}.pddl"
    sas_plan_file="sas_plan"

    # Check if files exist
    if [ ! -f "domain.pddl" ]; then
        echo "Error: domain.pddl not found!"
        exit 1
    fi
    if [ ! -f "$problem_file" ]; then
        echo "Error: Problem file $problem_file not found!"
        exit 1
    fi

    # Run Fast Downward
    echo "Running Fast Downward for level $level..."
    $FAST_DOWNWARD domain.pddl "$problem_file" --search "astar(lmcut())"

    # Check if the plan file exists
    if [ -f "$sas_plan_file" ]; then
        echo "Plan found for level $level."

        # Read the plan, exclude comments, and format each line as a single string
        plan_json=$(grep -v "^;" "$sas_plan_file" | sed 's/[()]//g' | sed 's/^/"/;s/$/"/' | tr '\n' ',' | sed 's/,$//')

        # Wrap the plan in square brackets to form a JSON array
        plan_json="[${plan_json}]"

        # Clean up sas_plan file after reading
        rm -f "$sas_plan_file"
    else
        echo "No plan found for level $level."
        plan_json="[]"
    fi

    # Add the level and plan to the JSON
    if [ $level -eq 13 ]; then
        # Last level (no trailing comma)
        echo "    \"$level\": $plan_json" >> "$output_file"
    else
        echo "    \"$level\": $plan_json," >> "$output_file"
    fi
done

# Close the JSON file
echo "}" >> "$output_file"

echo "Plans written to $output_file"
