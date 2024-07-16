import numpy as np
import pandas as pd
import os

def create_synthetic_data(output_dir, num_samples, image_shape, metadata_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create synthetic images
    images = np.random.rand(num_samples, *image_shape).astype(np.float32)

    # Save images to .npy file
    data_path = os.path.join(output_dir, 'data.npy')
    np.save(data_path, images)

    # Create synthetic metadata
    ages = np.random.randint(50, 90, size=num_samples)
    diagnoses = np.random.randint(0, 2, size=num_samples)  # 0 or 1

    metadata = pd.DataFrame({
        'id': np.arange(num_samples),
        'age': ages,
        'stroke': diagnoses
    })

    # Save metadata to CSV file
    metadata.to_csv(metadata_path, index=False)

    return data_path, metadata_path

# Parameters
output_dir = 'synthetic_dataset'
num_samples = 20
image_shape = (1, 121, 145, 121)
metadata_path = os.path.join(output_dir, 'metadata.csv')

# Create synthetic dataset
data_path, metadata_path = create_synthetic_data(output_dir, num_samples, image_shape, metadata_path)
print(f"Synthetic data saved to {data_path}")
print(f"Synthetic metadata saved to {metadata_path}")