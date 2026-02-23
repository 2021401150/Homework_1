'''import numpy as np
import pandas as pd

# 1. Load the data
data = np.load('First_deliverable_coordinat_predictions.npz')

# 2. Create a DataFrame
# We flatten() the arrays just in case they are multidimensional
df = pd.DataFrame({
    'Target': data['targets'].flatten(),
    'Prediction': data['predictions'].flatten()
})

# 3. View the first few rows
print(df.head())

# 4. Optional: Save to an actual CSV file to open in Excel/Sheets
df.to_csv('First_deliverable_results.csv', index=False)
print("\nResults also saved to 'First_deliverable_results.csv'")'''


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



'''
# Load the file
data2 = np.load('First_deliverable_outputs/First_deliverable_predictions.npz')

# List the keys inside (should be ['predictions', 'targets'])
print("Keys in file:", data2.files)

# Look at the first 5 guesses vs reality
preds = data2['predictions']
actual = data2['targets']

for i in range(5):
    print(f"Sample {i}: Model guessed {preds[i]}, Actual was {actual[i]}")

# 4. Save to CSV inside the output folder

df2 = pd.DataFrame({
    'True Value': actual[:, 0],
    'Predicted Value': preds[:, 0],
    
})

csv_save_path = 'First_deliverable_outputs/First_deliverable_predictions_results.csv'
df2.to_csv(csv_save_path, index=False)
print(f"\nResults successfully saved to '{csv_save_path}'")'''
    
    
