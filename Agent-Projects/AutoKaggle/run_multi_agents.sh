#!/bin/bash

competitions=("titanic" "spaceship_titanic" "ghouls_goblins_and_ghosts_boo" "house_prices" \
                "classification_with_an_academic_success_dataset" "bank_churn" "obesity_risks" "plate_defect")
start_run=1
end_run=5
dest_dir_param="all_tools"
model="gpt_4o"

run_experiment() {
    competition=$1
    run_number=$2

    echo "Running $competition - Experiment $run_number"

    # Run framework.py
    python framework.py --competition "$competition" --model "$model"

    # Define source and destination directories
    source_dir="multi_agents/competition/$competition"
    dest_dir="multi_agents/experiments_history/$competition/$model/$dest_dir_param/$run_number"

    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"

    # Move files and directories
    for item in "$source_dir"/*; do
        filename=$(basename "$item")
        if [[ "$filename" != "overview.txt" && "$filename" != "sample_submission.csv" && "$filename" != "test.csv" && "$filename" != "train.csv" && "$filename" != "data_description.txt" ]]; then
            if [ -d "$item" ]; then
                # If it's a directory, copy it recursively
                cp -R "$item" "$dest_dir/"
                rm -rf "$item"
            else
                # If it's a file, move it
                mv "$item" "$dest_dir/"
            fi
        fi
    done
}

# Run experiments for each competition
for competition in "${competitions[@]}"; do
    for ((run=start_run; run<=end_run; run++)); do
        run_experiment "$competition" "$run" "$model" "$dest_dir_param"
    done
done

echo "All experiments completed."
