import numpy as np
import pandas as pd
import os

selects=['First','Second']

for select in selects:

    # 1. Load the data from the correct output folder
    # Adjust the path if you are running this from outside the 'First_deliverable_outputs' folder
    file_path = select+'_deliverable_outputs/'+select+'_deliverable_coordinate_predictions.npz'

    if os.path.exists(file_path):
        data = np.load(file_path)
        
        # 2. Create a structured DataFrame
        # Assuming the data is shape (N, 2) for X and Y coordinates
        initial = data['true_pos']
        predicted = data['predicted_pos']
        
        df = pd.DataFrame({
            'True_X': initial[:, 0],
            'True_Y': initial[:, 1],
            'Predicted_X': predicted[:, 0],
            'Predicted_Y': predicted[:, 1]
        })

        # Calculate the predicted displacement for curiosity
        df['Predicted_Dist'] = np.sqrt((df['Predicted_X'] - df['True_X'])**2 + 
                                    (df['Predicted_Y'] - df['True_Y'])**2)

        # 3. View the first few rows
        print("Sample of Coordinate Predictions:")
        print(df.head())

        # 4. Save to CSV inside the output folder
        csv_save_path = select+'_deliverable_outputs/'+select+'_deliverable_coordinate_results.csv'
        df.to_csv(csv_save_path, index=False)
        print(f"\nResults successfully saved to '{csv_save_path}'")
    else:
        print(f"Error: Could not find {file_path}. Did you run the training script first?")


