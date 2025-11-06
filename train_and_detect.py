import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Conv2DTranspose, concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# --- Constants ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1 # We will force grayscale
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# --- Anomaly Thresholds (OPTIMIZED FOR CURRENT MODEL) ---
# Based on test results: Good Product - MSE: 0.002814, SSIM: 0.872220
#                        Bad Product  - MSE: 0.003891, SSIM: 0.820883
#
# PERFECTLY TUNED THRESHOLDS FOR MAXIMUM ACCURACY:
SSIM_THRESHOLD = 0.846 # If SSIM is *below* this, it's an anomaly.
MSE_THRESHOLD = 0.0033 # If MSE is *above* this, it's an anomaly.

# --- 1. Data Loading and Preparation ---

def load_and_prep_image(filepath, is_anomaly=False):
    """
    Loads one image, converts to grayscale, resizes, and normalizes.
    """
    try:
        # Read in grayscale mode
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: Could not read image at {filepath}")
            return None
            
        # Resize to our new, higher resolution
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Save a debug copy so you can see what the model sees
        if not is_anomaly:
            cv2.imwrite(f"images/debug_resized_good.png", img)
        else:
            cv2.imwrite(f"images/debug_resized_bad.png", img)

        # Normalize (0-255 -> 0.0-1.0) and reshape
        img = img.astype('float32') / 255.
        img = np.reshape(img, (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        return img
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_training_data_from_base(base_good_image, samples=1000):
    """
    *** CRITICAL CHANGE ***
    Creates two arrays:
    1. x_train: "Dirty" images (augmented with noise/rotations)
    2. y_train: "Clean" images (the original, perfect image)
    The model will learn to turn 'x' into 'y'.
    """
    print(f"Creating {samples} 'dirty' (x) and 'clean' (y) training images...")
    x_train_dirty = []
    y_train_clean = []
    
    base_img_squeezed_8bit = (np.squeeze(base_good_image) * 255.).astype(np.uint8)
    base_img_normalized_32f = np.squeeze(base_good_image).astype(np.float32)

    for _ in range(samples):
        aug_img = base_img_squeezed_8bit.copy()
        
        # 1. Add VERY subtle noise
        noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8) # Increased noise slightly
        aug_img = cv2.add(aug_img, noise)

        # 2. Add VERY subtle rotation
        angle = np.random.uniform(-5, 5) # Increased rotation slightly
        M_rot = cv2.getRotationMatrix2D((IMG_WIDTH/2, IMG_HEIGHT/2), angle, 1)
        aug_img = cv2.warpAffine(aug_img, M_rot, (IMG_WIDTH, IMG_HEIGHT), borderValue=0)
        
        # 3. Add VERY subtle shift
        shift_x = np.random.uniform(-2, 2) # Increased shift slightly
        shift_y = np.random.uniform(-2, 2)
        M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aug_img = cv2.warpAffine(aug_img, M_trans, (IMG_WIDTH, IMG_HEIGHT), borderValue=0)

        # 4. Brightness adjustment
        hsv = cv2.cvtColor(cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], np.random.randint(-10, 10))
        aug_img = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        # Add the "dirty" image to x_train
        x_train_dirty.append(aug_img)
        # Add the "perfect" original image to y_train
        y_train_clean.append(base_img_normalized_32f)

    # Normalize x_train
    x_train_dirty = np.array(x_train_dirty).astype('float32') / 255.
    x_train_dirty = np.reshape(x_train_dirty, (len(x_train_dirty), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # y_train is already normalized, just need to reshape
    y_train_clean = np.array(y_train_clean)
    y_train_clean = np.reshape(y_train_clean, (len(y_train_clean), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    print("Augmented training data created (x_train, y_train).")
    return x_train_dirty, y_train_clean

# --- 2. Build The PRO "U-Net" Model ---

def build_unet_autoencoder(shape):
    """
    Builds a DEEPER U-Net style Autoencoder with Dropout.
    This will increase separation between good/bad scores.
    """
    print("Building FINAL Deeper U-Net Autoencoder...")
    inputs = Input(shape=shape)

    # --- Encoder Path (Downsampling) ---
    # 128x128x1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) # Start with 64 filters
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1) # 64x64x64
    p1 = Dropout(0.1)(p1) # Add Dropout

    # 64x64x64
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2) # 32x32x128
    p2 = Dropout(0.1)(p2)

    # 32x32x128
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3) # 16x16x256
    p3 = Dropout(0.1)(p3)

    # --- Bottleneck ---
    # 16x16x256
    b = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    b = BatchNormalization()(b)
    b = Conv2D(512, (3, 3), activation='relu', padding='same')(b) # 16x16x512

    # --- Decoder Path (Upsampling) ---
    # 16x16x512
    u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b) # 32x32x256
    u3 = concatenate([u3, c3]) 
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    # 32x32x256
    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6) # 64x64x128
    u2 = concatenate([u2, c2])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    # 64x64x128
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7) # 128x128x64
    u1 = concatenate([u1, c1])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    # Use 'sigmoid' activation, but we will change the loss function
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    # *** CRITICAL CHANGE: Use 'mse' loss ***
    # This will directly optimize for Mean Squared Error
    model.compile(optimizer='adam', loss='mse')
    
    print("Model built successfully.")
    model.summary()
    return model

# --- 3. Utility Functions ---

def compare_images(img1, img2):
    """
    Calculates Mean Squared Error (MSE) and Structural Similarity (SSIM)
    """
    # .squeeze() removes the extra dimensions (like batch and channel)
    img1_sq = np.squeeze(img1)
    img2_sq = np.squeeze(img2)
    
    mse = mean_squared_error(img1_sq, img2_sq)
    
    # data_range is 1.0 because we normalized pixels to be between 0 and 1
    ssim_score = ssim(img1_sq, img2_sq, data_range=1.0) 
    return mse, ssim_score

def visualize_prediction(original, reconstructed, mse, ssim_score, title_prefix=""):
    """
    Displays the Original, Reconstructed, and Heatmap images.
    """
    # --- This is the "Heatmap" ---
    # We calculate the absolute difference between the two images
    diff = np.abs(original - reconstructed)
    diff = np.squeeze(diff) # Remove extra dims
    
    # Squeeze images for display
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)

    # Plot the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.set_axis_off()
    
    ax2.imshow(reconstructed, cmap='gray')
    ax2.set_title('Reconstructed Image')
    ax2.set_axis_off()
    
    # Use 'jet' colormap for the heatmap.
    # Set vmin/vmax to fix the color scale, making anomalies "pop"
    heatmap_plot = ax3.imshow(diff, cmap='jet', vmin=0, vmax=1.0) 
    ax3.set_title('Anomaly Heatmap')
    ax3.set_axis_off()
    
    # Add a color bar to the heatmap
    fig.colorbar(heatmap_plot, ax=ax3, orientation='vertical')
    
    # Check against thresholds
    is_anomaly_mse = mse > MSE_THRESHOLD
    is_anomaly_ssim = ssim_score < SSIM_THRESHOLD
    
    status = "PRODUCT OK"
    if is_anomaly_mse or is_anomaly_ssim:
        # We can add which rule it broke
        status = f"ANOMALY DETECTED (MSE: {is_anomaly_mse}, SSIM: {is_anomaly_ssim})"

    fig.suptitle(f"{title_prefix} | Status: {status}\n(MSE: {mse:.6f} | SSIM: {ssim_score:.6f})", fontsize=16)
    
    # Save the figure
    fig_filename = f"images/result_{title_prefix.replace(' ', '_').replace(':', '')}.png"
    plt.savefig(fig_filename)
    print(f"Saved result image to {fig_filename}")
    
    plt.show() # Also display it

# --- 4. Main Execution ---

def find_image_path(base_name):
    """
    Finds image with common extensions and validates it's an image.
    """
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        path = os.path.join("images", base_name + ext)
        if os.path.exists(path):
            # Check if it's a valid image file using cv2
            try:
                test_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if test_img is not None:
                    return path
                else:
                    print(f"Found {path}, but it's not a valid image file. Skipping.")
            except:
                print(f"Found {path}, but it's not a valid image file. Skipping.")
    return None

if __name__ == "__main__":
    
    # --- Step 1: Load your custom images ---
    print("--- LOADING CUSTOM IMAGES (128x128) ---")
    
    good_img_path = find_image_path("demo_good")
    bad_img_path = find_image_path("demo_bad")

    if not good_img_path:
        print("FATAL ERROR: Could not find 'demo_good' (png, jpg, etc.) in 'images/' folder.")
        exit()
    if not bad_img_path:
        print("FATAL ERROR: Could not find 'demo_bad' (png, jpg, etc.) in 'images/' folder.")
        exit()
        
    print(f"Found 'good' image: {good_img_path}")
    print(f"Found 'bad' image: {bad_img_path}")

    base_good_image = load_and_prep_image(good_img_path)
    anomaly_image = load_and_prep_image(bad_img_path, is_anomaly=True)
    
    if base_good_image is None or anomaly_image is None:
        print("Exiting due to image loading errors.")
        exit()
        
    print("\nCheck 'images/debug_resized_good.png' and 'debug_resized_bad.png'.")
    print("If they look OK, training will begin.")

    # --- Step 2: Create training data (x=dirty, y=clean) ---
    x_train, y_train = create_training_data_from_base(base_good_image, samples=1000)
    
    # --- Step 3: Build the FINAL model ---
    autoencoder = build_unet_autoencoder(IMG_SHAPE)
    
    # --- Step 4: Train the model ---
    print("\n--- STARTING MODEL TRAINING (DENOISING U-NET) ---")
    print("This will take a 5-10 minutes. Please wait...")
    
    early_stopper = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Increased patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    # *** CRITICAL CHANGE: Fit (x_train, y_train) ***
    autoencoder.fit(
        x_train, # The "dirty" images
        y_train, # The "clean" images
        epochs=100, 
        batch_size=16, 
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stopper, reduce_lr]
    )
    print("--- MODEL TRAINING COMPLETE ---")
    
    # --- Step 5: Run the Demo! ---
    print("\n--- RUNNING VULCAN-SHIELD FINAL DEMO ---")
    
    # DEMO 1: "PERFECT" PRODUCT
    print("\n[TEST 1] Analyzing your 'PERFECT' product ('demo_good.jpg')...")
    reconstructed_good = autoencoder.predict(base_good_image)
    mse_good, ssim_good = compare_images(base_good_image, reconstructed_good)
    visualize_prediction(base_good_image, reconstructed_good, mse_good, ssim_good, title_prefix="[TEST 1 PERFECT PRODUCT]")

    # DEMO 2: "ANOMALY" PRODUCT
    print("\n[TEST 2] Analyzing your 'ANOMALY' product ('demo_bad.jpg')...")
    reconstructed_bad = autoencoder.predict(anomaly_image)
    mse_bad, ssim_bad = compare_images(anomaly_image, reconstructed_bad)
    visualize_prediction(anomaly_image, reconstructed_bad, mse_bad, ssim_bad, title_prefix="[TEST 2 ANOMALY PRODUCT]")
    
    print("\n--- DEMO COMPLETE ---")
    print("\n\n=== ACTION REQUIRED: THRESHOLD TUNING ===")
    print("The model has run. Now you MUST calibrate it.")
    print("Look at your scores below, then edit the thresholds at the top of this script.")
    print("\n******************************************************************")
    print(f"     'Good' Product Scores -> MSE: {mse_good:.6f} | SSIM: {ssim_good:.6f}")
    print(f"      'Bad' Product Scores -> MSE: {mse_bad:.6f} | SSIM: {ssim_bad:.6f}")
    print("******************************************************************\n")
    print("Example: If 'Good' SSIM is 0.95 and 'Bad' SSIM is 0.70, a good SSIM_THRESHOLD is 0.85")
    print("Example: If 'Good' MSE is 0.005 and 'Bad' MSE is 0.030, a good MSE_THRESHOLD is 0.015")
    print("\n---> RE-RUN THE SCRIPT AFTER UPDATING THE THRESHOLDS! <---")