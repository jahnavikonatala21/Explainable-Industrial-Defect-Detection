"""
Train Defect Detection Model using MVTec AD Dataset
Trains on your custom dataset at data/defect_detection
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shutil
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from config.config import *

# ===============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ===============================================
MVTEC_DIR = os.path.join(BASE_DIR, 'data', 'defect_detection')

# Select which products to train on (or 'all' for everything)
PRODUCTS_TO_TRAIN = 'all'

EPOCHS_CUSTOM = 50  # More epochs for better learning
BATCH_SIZE_CUSTOM = 32  # Larger batch for stability  
LEARNING_RATE_CUSTOM = 0.001  # Higher initial LR with scheduler
FINE_TUNE_EPOCHS = 30  # Additional epochs for fine-tuning
FINE_TUNE_LAYERS = 50  # Number of MobileNetV2 layers to unfreeze

# ===============================================
# SIMPLIFIED CLASS MAPPING (6 CATEGORIES)
# ===============================================
SIMPLIFIED_CLASSES = ['no_defect', 'defect', 'contamination', 'scratch', 'bend', 'cut']

def map_to_simplified_class(original_class):
    """Map original 49 classes to 6 simplified categories"""
    original_class = original_class.lower()
    
    # 1. No defect
    if original_class in ['no_defect', 'good']:
        return 'no_defect'
    
    # 2. Contamination (includes all contamination-related defects)
    contamination_keywords = ['contamination', 'oil', 'liquid', 'glue', 'color', 'gray_stroke', 'print', 'faulty_imprint']
    for keyword in contamination_keywords:
        if keyword in original_class:
            return 'contamination'
    
    # 3. Scratch (includes all scratch-related defects)
    scratch_keywords = ['scratch', 'rough', 'poke', 'hole', 'thread']
    for keyword in scratch_keywords:
        if keyword in original_class:
            return 'scratch'
    
    # 4. Bend (includes all bending/deformation defects)
    bend_keywords = ['bent', 'fold', 'squeeze', 'flip', 'misplaced', 'damaged']
    for keyword in bend_keywords:
        if keyword in original_class:
            return 'bend'
    
    # 5. Cut (includes all cut/broken defects)
    cut_keywords = ['cut', 'broken', 'crack', 'split', 'missing', 'cable_swap']
    for keyword in cut_keywords:
        if keyword in original_class:
            return 'cut'
    
    # 6. General defect (anything else)
    return 'defect'

# ===============================================
# DATA PREPARATION
# ===============================================

def get_mvtec_products():
    """Get all available products in MVTec dataset"""
    products = []
    for item in os.listdir(MVTEC_DIR):
        product_path = os.path.join(MVTEC_DIR, item)
        if os.path.isdir(product_path) and os.path.exists(os.path.join(product_path, 'train')):
            products.append(item)
    return products

def get_defect_types(product):
    """Get all defect types for a product"""
    test_dir = os.path.join(MVTEC_DIR, product, 'test')
    if not os.path.exists(test_dir):
        return []
    
    defect_types = []
    for item in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, item)):
            defect_types.append(item)
    return defect_types

def prepare_training_data():
    """Prepare training data from MVTec dataset with SIMPLIFIED 6 CLASSES"""
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA - 6 SIMPLIFIED CLASSES")
    print("="*70)
    print(f"Target classes: {SIMPLIFIED_CLASSES}")
    
    all_products = get_mvtec_products()
    print(f"\nAvailable products: {all_products}")
    
    if PRODUCTS_TO_TRAIN == 'all':
        products = all_products
    else:
        products = [p for p in PRODUCTS_TO_TRAIN if p in all_products]
    
    print(f"Training on products: {products}")
    
    # Collect all images with simplified labels
    images = []
    labels = []
    original_to_simplified = {}  # Track mapping for logging
    
    for product in products:
        print(f"\nProcessing {product}...")
        
        # Get good images from train folder
        good_dir = os.path.join(MVTEC_DIR, product, 'train', 'good')
        if os.path.exists(good_dir):
            for img_file in os.listdir(good_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(good_dir, img_file)
                    images.append(img_path)
                    labels.append('no_defect')
        
        # Get defective images from test folder
        test_dir = os.path.join(MVTEC_DIR, product, 'test')
        if os.path.exists(test_dir):
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if os.path.isdir(defect_dir) and defect_type != 'good':
                    # Map to simplified class
                    simplified_class = map_to_simplified_class(defect_type)
                    if defect_type not in original_to_simplified:
                        original_to_simplified[defect_type] = simplified_class
                    
                    for img_file in os.listdir(defect_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(defect_dir, img_file)
                            images.append(img_path)
                            labels.append(simplified_class)
    
    # Use the predefined simplified classes
    class_names = SIMPLIFIED_CLASSES
    
    print(f"\n{'='*50}")
    print("CLASS MAPPING:")
    print(f"{'='*50}")
    for orig, simp in sorted(original_to_simplified.items()):
        print(f"  {orig:25} -> {simp}")
    
    print(f"\n{'='*50}")
    print(f"Total images: {len(images)}")
    print(f"Classes: {class_names} ({len(class_names)} classes)")
    
    # Count per class
    from collections import Counter
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for cls in class_names:
        print(f"  {cls}: {label_counts.get(cls, 0)} images")
    
    return images, labels, class_names

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = keras.preprocessing.image.load_img(
        img_path, 
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

def create_dataset(images, labels, class_names):
    """Create TensorFlow dataset from images and labels"""
    print("\nLoading and preprocessing images...")
    
    X = []
    y = []
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for i, (img_path, label) in enumerate(zip(images, labels)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(images)} images...")
        
        try:
            img = load_and_preprocess_image(img_path)
            X.append(img)
            y.append(class_to_idx[label])
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert to one-hot encoding
    y = keras.utils.to_categorical(y, num_classes=len(class_names))
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y

# ===============================================
# MODEL BUILDING
# ===============================================

def build_model(num_classes):
    """Build the defect detection model using Functional API (GradCAM compatible)"""
    print("\nBuilding model with Functional API (GradCAM compatible)...")
    
    # Create input layer
    inputs = keras.Input(shape=INPUT_SHAPE)
    
    # Load base model
    base_model = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    
    # Initially freeze all base model layers
    base_model.trainable = False
    
    # Build the model using Functional API
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)  # Stronger dropout
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_CUSTOM),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built with Functional API - GradCAM will work!")
    
    return model, base_model

# ===============================================
# TRAINING
# ===============================================

def train_model():
    """Main training function with two-phase training for >90% accuracy"""
    print("\n" + "="*70)
    print("DEFECT DETECTION MODEL TRAINING (OPTIMIZED FOR >90% ACCURACY)")
    print("="*70)
    
    # Prepare data
    images, labels, class_names = prepare_training_data()
    
    if len(images) == 0:
        print("ERROR: No images found!")
        return
    
    # Create dataset
    X, y = create_dataset(images, labels, class_names)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"\nTraining set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = y_train.argmax(axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights computed for {len(class_weight_dict)} classes")
    
    # Build model
    model, base_model = build_model(len(class_names))
    model.summary()

    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # ========== PHASE 1: Train head only (frozen backbone) ==========
    print("\n" + "="*70)
    print("PHASE 1: Training head layers (backbone frozen)...")
    print("="*70)
    
    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE_CUSTOM),
        validation_data=(X_val, y_val),
        epochs=EPOCHS_CUSTOM,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Get Phase 1 accuracy
    phase1_acc = max(history1.history['val_accuracy'])
    print(f"\n✅ Phase 1 Complete - Best Validation Accuracy: {phase1_acc:.2%}")
    
    # ========== PHASE 2: Fine-tune backbone ==========
    print("\n" + "="*70)
    print(f"PHASE 2: Fine-tuning top {FINE_TUNE_LAYERS} layers of MobileNetV2...")
    print("="*70)
    
    # Unfreeze the top layers of the backbone
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_CUSTOM / 10),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Trainable layers: {sum(1 for l in model.layers if l.trainable)}")
    
    # Continue training with fine-tuning
    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE_CUSTOM),
        validation_data=(X_val, y_val),
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Get Phase 2 accuracy
    phase2_acc = max(history2.history['val_accuracy'])
    print(f"\n✅ Phase 2 Complete - Best Validation Accuracy: {phase2_acc:.2%}")
    
    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Model saved to: {FINAL_MODEL_PATH}")
    
    # Save class names
    class_names_path = os.path.join(MODELS_DIR, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    print(f"✅ Class names saved to: {class_names_path}")
    
    # Update config with new class names
    update_config_classes(class_names)
    
    # Final evaluation
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.2%}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    return model, history2, class_names

def update_config_classes(class_names):
    """Update config.py with new class names"""
    config_path = os.path.join(BASE_DIR, 'config', 'config.py')
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update CLASS_NAMES
    old_line = None
    for line in content.split('\n'):
        if line.startswith('CLASS_NAMES'):
            old_line = line
            break
    
    if old_line:
        new_line = f"CLASS_NAMES = {class_names}"
        content = content.replace(old_line, new_line)
        
        # Update NUM_CLASSES
        content = content.replace(
            'NUM_CLASSES = len(CLASS_NAMES)',
            f'NUM_CLASSES = {len(class_names)}  # Updated automatically'
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated config.py with {len(class_names)} classes")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MVTec AD DEFECT DETECTION TRAINING")
    print("="*70)
    print(f"\nDataset location: {MVTEC_DIR}")
    print(f"Products to train: {PRODUCTS_TO_TRAIN}")
    print(f"Epochs: {EPOCHS_CUSTOM}")
    print(f"Batch size: {BATCH_SIZE_CUSTOM}")
    print(f"Learning rate: {LEARNING_RATE_CUSTOM}")
    print("="*70)
    
    input("\nPress Enter to start training... (Ctrl+C to cancel)")
    
    model, history, class_names = train_model()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTrained defect types: {class_names}")
    print(f"Model saved to: {CHECKPOINT_PATH}")
    print("\nNow you can run the app to detect these defects with hot-spot heatmaps!")
