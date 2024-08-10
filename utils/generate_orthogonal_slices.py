import nrrd
import numpy as np
import matplotlib.pyplot as plt

def load_mri_data(filepath):
    """Load MRI data from a NIfTI file."""
    img, header = nrrd.read(filepath)
    
    return img

def get_orthogonal_slices(volume):
    """Extract orthogonal slices from a 3D MRI volume."""
    # Axial slice (along the z-axis)
    axial_slice = volume[:, :, volume.shape[2] // 2]
    
    # Coronal slice (along the y-axis)
    coronal_slice = volume[:, volume.shape[1] // 2, :]
    
    # Sagittal slice (along the x-axis)
    sagittal_slice = volume[volume.shape[0] // 2, :, :]
    
    return axial_slice, coronal_slice, sagittal_slice

def plot_slices(axial, coronal, sagittal):
    """Plot the orthogonal slices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot axial slice
    axes[0].imshow(axial.T, cmap='gray', origin='lower')
    axes[0].set_title('Axial Slice')
    axes[0].axis('off')

    # Plot coronal slice
    axes[1].imshow(coronal.T, cmap='gray', origin='lower')
    axes[1].set_title('Coronal Slice')
    axes[1].axis('off')

    # Plot sagittal slice
    axes[2].imshow(sagittal.T, cmap='gray', origin='lower')
    axes[2].set_title('Sagittal Slice')
    axes[2].axis('off')

    plt.show()

