import argparse
import glob
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    AsDiscreted,
    Invertd,
    SaveImaged,
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import SwinUNETR


def save_2nd_channel(output_nifti_path, predictions, subject_id):
    # Make sure the predictions are 4D
    if len(predictions.shape) != 4:
        raise ValueError("The input predictions should be 4D.")

    # Convert predictions to a NumPy array
    predictions_np = predictions.detach().cpu().numpy()

    # Extract the second channel (assuming zero-based indexing)
    second_channel = predictions_np[1, :, :, :]

    # Define the transformation
    transformation = np.array([[1., 0., 0., -96.],
                               [0., 1., 0., -132.],
                               [0., 0., 1., -78.],
                               [0., 0., 0., 1.]])

    # Create a new NIfTI image using the second channel data and the transformation
    new_img = nib.Nifti1Image(second_channel, transformation)

    # Save the new NIfTI image in .nii.gz format
    subject_output_dir = os.path.join(os.path.dirname(output_nifti_path), f"{subject_id}_ct")
    os.makedirs(subject_output_dir, exist_ok=True)
    output_nifti_filename = os.path.basename(output_nifti_path)
    output_nifti_path = os.path.join(subject_output_dir, output_nifti_filename)
    nib.save(new_img, output_nifti_path)


def main(input_dir, output_dir):
    device = torch.device("cuda:0")
    num_heads = 10  # 12 normally
    embed_dim = 512  # 768 normally
    roi_size = [128, 128, 64]
    pixdim = (1.5, 1.5, 2.0)

    model = SwinUNETR(
        img_size=roi_size,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        norm_name='batch',
    ).to(device)

    #    model = UNet(
    #        spatial_dims=3,
    #        in_channels=1,
    #        out_channels=2,
    #        channels=(16, 32, 64, 128, 256),
    #        strides=(2, 2, 2, 2),
    #        num_res_units=2,
    #        norm="batch",  # using batch normalization
    #    ).to('cuda')

    # Load the trained model
    model_path = os.path.join(os.getcwd(), "SWIN_UNETR_48_100_best_metric_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')  # device can be 'cuda' or 'cpu'
    model.eval()

    # Search for "*_ct.nii.gz" files directly in input_dir
    print(f"Searching in directory: {input_dir}")  # Debug print
    files = glob.glob(os.path.join(input_dir, "*_ct.nii.gz"))
    print(f"Found files: {files}")  # Debug print
    test_images = files
    print(f"Test images: {test_images}")

    # Create a list of dictionaries where each dictionary contains a single key-value pair.
    # The key is "image", and the value is the file path to a test image.
    test_data = [{"image": image} for image in test_images]

    # Define the transforms to be applied before inference
    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    # Create a Dataset and DataLoader for the test data
    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)

    print(f"Number of batches in test_org_loader: {len(test_org_loader)}")

    # Define the post-transforms to be applied after inference
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg",
                       resample=True),
        ]
    )

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist. Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Create 'segmentations' subfolder in output directory
    segmentations_dir = os.path.join(output_dir, 'segmentations')
    os.makedirs(segmentations_dir, exist_ok=True)

    # Perform inference on the test data
    with torch.no_grad():
        for test_data in test_org_loader:
            # Get the input images
            test_images = test_data["image"].to('cuda')

            # Perform inference
            roi_size = (128, 128, 64)  # Adjust this to match your network input size
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_images, roi_size, sw_batch_size, model)

            # Apply the post-transforms
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            # Save the second channel as a new .nii.gz image
            input_nifti_path = test_data[0]["image_meta_dict"]["filename_or_obj"]
            subject_id = os.path.basename(os.path.dirname(input_nifti_path))
            output_nifti_path = os.path.join(segmentations_dir,
                                             f"second_channel_{os.path.basename(input_nifti_path)}")  # Save in 'segmentations' subfolder
            predictions = test_data[0]["pred"]
            save_2nd_channel(output_nifti_path, predictions, subject_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform hemorrhage segmentation on NIfTI images.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing the input NIfTI images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where the output NIfTI images will be saved.')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
