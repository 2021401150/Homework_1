import numpy as np
from homework1 import Hw1Env

def collect_data(num_samples=1000, save_path='robot_data.npz'):
    print(f"Starting data collection for {num_samples} samples...")
    # Using "headless" (if supported) or "gui" 
    env = Hw1Env(render_mode="offscreen") 
    
    # Initialize lists to hold our data
    data_img_before = []
    data_action = []
    data_pos_after = []
    data_img_after = []
    
    for i in range(num_samples):
        env.reset()
        action_id = np.random.randint(4) # 4 possible directions
        
        # Get state before action
        _, img_before = env.state()
        
        # Take the action
        env.step(action_id)
        
        # Get state after action
        pos_after, img_after = env.state()
        
        # Store in our lists
        data_img_before.append(img_before)
        data_action.append(action_id)
        data_pos_after.append(pos_after)
        data_img_after.append(img_after)
        
        # Print progress so you know it hasn't crashed
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1}/{num_samples} samples")
            
    # Clean up environment
    env.reset() 
    
    # Save all arrays to a single compressed file
    np.savez_compressed(
        save_path, 
        img_before=np.array(data_img_before),
        action=np.array(data_action),
        pos_after=np.array(data_pos_after),
        img_after=np.array(data_img_after)
    )
    print(f"Data successfully saved to {save_path}")

if __name__ == "__main__":
    # The assignment asks for 1000 data points
    collect_data(1000)