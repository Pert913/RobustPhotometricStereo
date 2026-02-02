"""
End-to-End Pipeline: Predict Light Directions + Run RPS
========================================================
This script demonstrates the complete workflow:
1. Train light direction predictor (or load existing)
2. Predict light directions for your own images
3. Run RPS to get surface normals

Updated for LightDirectionPredictor with RF model.
"""

from __future__ import print_function
import numpy as np
import os
import sys
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rps import RPS
import ps_utils
from light_direction_predictor import LightDirectionPredictor


def train_light_predictor(training_folder, save_path, model_type='rf'):
    """
    Train the light direction predictor from scratch.
    """
    print("=" * 60)
    print("STEP 1: Training Light Direction Predictor")
    print("=" * 60)

    predictor = LightDirectionPredictor(
        img_size=64,
        n_pca=64,
        normalize_images=True
    )

    # Load and prepare training data
    X, Y, groups, _ = predictor.load_training_data(training_folder)

    # Cross-validation to assess generalization
    print("\n--- Cross-Validation Results ---")
    predictor.cross_validate(X, Y, groups, model_type=model_type)

    # Train final model
    print("\n--- Training Final Model ---")
    predictor.train(X, Y, model_type=model_type)

    # Save
    predictor.save(save_path)

    return predictor


def get_base_paths():
    """Get base paths for the project."""
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    SCRIPT_DIR = os.path.dirname(__file__)
    return BASE_DIR, SCRIPT_DIR


def predict_light_directions(predictor, images_folder, output_txt_path):
    """
    Predict light directions for images in a folder.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Predicting Light Directions")
    print("=" * 60)
    light_dirs = predictor.predict_from_folder(images_folder, output_txt_path)
    print(f"\nPredicted {len(light_dirs)} light directions")
    print(f"Sample predictions (first 5):")
    for i in range(min(5, len(light_dirs))):
        print(f"  Image {i + 1}: [{light_dirs[i, 0]:.4f}, {light_dirs[i, 1]:.4f}, {light_dirs[i, 2]:.4f}]")
    return light_dirs


def run_rps_pipeline(images_folder, light_txt_path, mask_path, output_normal_path,
                     method=RPS.L2_SOLVER, use_npy=False):
    """
    Run the standard RPS pipeline.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Running Robust Photometric Stereo")
    print("=" * 60)

    rps = RPS()

    # Load mask
    if mask_path and os.path.exists(mask_path):
        rps.load_mask(filename=mask_path)
        print(f"Loaded mask: {mask_path}")
    else:
        print("No mask provided - using all pixels")

    # Load light directions
    rps.load_lighttxt(filename=light_txt_path)
    print(f"Loaded light directions: {light_txt_path}")

    # Load images
    if use_npy:
        rps.load_npyimages(foldername=images_folder)
    else:
        rps.load_images(foldername=images_folder, ext='png')
    print(f"Loaded images from: {images_folder}")

    # Solve
    method_names = {
        RPS.L2_SOLVER: "L2 (Least Squares)",
        RPS.L1_SOLVER: "L1 Residual Min",
        RPS.L1_SOLVER_MULTICORE: "L1 (Multicore)",
        RPS.SBL_SOLVER: "Sparse Bayesian Learning",
        RPS.SBL_SOLVER_MULTICORE: "SBL (Multicore)",
        RPS.RPCA_SOLVER: "Robust PCA"
    }
    print(f"Solving with method: {method_names.get(method, 'Unknown')}")

    start = time.time()
    rps.solve(method)
    elapsed = time.time() - start
    print(f"Computation time: {elapsed:.2f} seconds")

    # Save result
    rps.save_normalmap(filename=output_normal_path)
    print(f"Saved normal map: {output_normal_path}.npy")

    return rps


def evaluate_results(rps, gt_normal_path=None):
    """
    Evaluate and display results.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation & Visualization")
    print("=" * 60)

    if gt_normal_path and os.path.exists(gt_normal_path):
        N_gt = ps_utils.load_normalmap_from_npy(filename=gt_normal_path)
        N_gt = np.reshape(N_gt, (rps.height * rps.width, 3))

        angular_err = ps_utils.evaluate_angular_error(N_gt, rps.N, rps.background_ind)
        mean_err = np.mean(angular_err)

        print(f"Mean Angular Error: {mean_err:.2f} degrees")
    else:
        print("No ground truth available for quantitative evaluation")

    # Display normal map
    print("Displaying normal map (press any key to close)...")
    ps_utils.disp_normalmap(normal=rps.N, height=rps.height, width=rps.width)


# ==================== COMPLETE PIPELINE ====================

def full_pipeline_with_training():
    """
    Full pipeline: Train predictor, predict lights, run RPS.
    """
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

    TRAINING_FOLDER = os.path.join(BASE_DIR, 'data/training/')
    # Using ML model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'light_predictor_ml_v3.pkl')
    # Using DL model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'light_predictor_dl_v3.pkl')
    MODEL_TYPE = 'rf'  # Best performing model


    NEW_IMAGES_FOLDER = os.path.join(BASE_DIR, 'data/testing/images/')
    MASK_PATH = os.path.join(BASE_DIR, 'data/testing/mask.png')

    # Outputs
    PREDICTED_LIGHTS_PATH = os.path.join(BASE_DIR, 'data/testing/predicted_light_directions.txt')
    OUTPUT_NORMAL_PATH = os.path.join(BASE_DIR, 'data/testing/estimated_normal')

    RPS_METHOD = RPS.L2_SOLVER

    # === EXECUTION ===

    # Step 1: Train (or load existing model)
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        predictor = LightDirectionPredictor()
        predictor.load(MODEL_PATH)
    else:
        predictor = train_light_predictor(TRAINING_FOLDER, MODEL_PATH, MODEL_TYPE)

    # Step 2: Predict light directions
    predict_light_directions(predictor, NEW_IMAGES_FOLDER, PREDICTED_LIGHTS_PATH)

    # Step 3: Run RPS
    rps = run_rps_pipeline(
        images_folder=NEW_IMAGES_FOLDER,
        light_txt_path=PREDICTED_LIGHTS_PATH,
        mask_path=MASK_PATH,
        output_normal_path=OUTPUT_NORMAL_PATH,
        method=RPS_METHOD
    )

    # Step 4: Evaluate/Display
    evaluate_results(rps, gt_normal_path=None)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

# TODO: Assinged to Tue
def quick_test_on_real_images(test_object="bootaoPNG", test_folder="testing"):
    """
    Quick test: Use predictor on one of the real objects

    Args:
        test_object: Name of object folder to test (e.g., 'buddhaPNG', 'catPNG')
    """
    print("\n" + "=" * 60)
    print(f"QUICK TEST: Verify Pipeline on {test_object}")
    print("=" * 60)

    # Paths relative to project root
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    TESTING_FOLDER = os.path.join(os.path.dirname(__file__), 'data/testing/')

    # Model could be in script dir or project root
    MODEL_PATH_LOCAL = os.path.join(os.path.dirname(__file__), 'light_predictor_v2.pkl')
    MODEL_PATH_ROOT = os.path.join(BASE_DIR, 'light_predictor_v2.pkl')

    # Check which one exists
    if os.path.exists(MODEL_PATH_LOCAL):
        MODEL_PATH = MODEL_PATH_LOCAL
    elif os.path.exists(MODEL_PATH_ROOT):
        MODEL_PATH = MODEL_PATH_ROOT
    else:
        print(f"Model not found at:")
        print(f"  - {MODEL_PATH_LOCAL}")
        print(f"  - {MODEL_PATH_ROOT}")
        print("Please run light_direction_predictor.py first to train the model.")
        return

    predictor = LightDirectionPredictor()
    predictor.load(MODEL_PATH)

    # Test on specified object
    test_folder = os.path.join(TESTING_FOLDER, test_object)

    if not os.path.exists(test_folder):
        print(f"Test folder not found: {test_folder}")
        return

    gt_lights_path = os.path.join(test_folder, 'light_directions.txt')
    pred_lights_path = os.path.join(test_folder, 'predicted_light_directions.txt')

    # # Predict
    predicted_lights = predictor.predict_from_folder(test_folder, pred_lights_path)

    # Compare with ground truth
    gt_lights = np.loadtxt(gt_lights_path)

    # Angular error between predicted and GT light directions
    cos_sim = np.sum(gt_lights * predicted_lights, axis=1)
    cos_sim = np.clip(cos_sim, -1, 1)
    light_errors = np.arccos(cos_sim) * 180 / np.pi

    print(f"\nLight Direction Prediction Errors for {test_object}:")
    print(f"  Mean: {light_errors.mean():.2f}°")
    print(f"  Std:  {light_errors.std():.2f}°")
    print(f"  Min:  {light_errors.min():.2f}°")
    print(f"  Max:  {light_errors.max():.2f}°")

    # Now run RPS with predicted lights and compare normals
    print("\n--- Running RPS with Predicted Lights ---")
    mask_path = os.path.join(test_folder, '/metadata/mask.png')

    rps_pred = run_rps_pipeline(
        images_folder=test_folder + '/',
        light_txt_path=pred_lights_path,
        mask_path=mask_path if os.path.exists(mask_path) else None,
        output_normal_path=os.path.join(BASE_DIR, 'test_normal_predicted'),
        method=RPS.L2_SOLVER
    )

    print("\n--- Running RPS with Ground Truth Lights ---")
    rps_gt = run_rps_pipeline(
        images_folder=test_folder + '/',
         light_txt_path=gt_lights_path,
         mask_path=mask_path if os.path.exists(mask_path) else None,
         output_normal_path=os.path.join(BASE_DIR, 'test_normal_gt'),
         method=RPS.L2_SOLVER
    )

    # Compare normal maps
    normal_cos_sim = np.sum(rps_gt.N * rps_pred.N, axis=1)
    normal_cos_sim = np.clip(normal_cos_sim, -1, 1)
    normal_errors = np.arccos(normal_cos_sim) * 180 / np.pi

    if rps_pred.background_ind is not None:
         normal_errors[rps_pred.background_ind] = 0
         foreground_mask = np.ones(len(normal_errors), dtype=bool)
         foreground_mask[rps_pred.background_ind] = False
         mean_normal_err = normal_errors[foreground_mask].mean()
    else:
         mean_normal_err = normal_errors.mean()

    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print('=' * 50)
    print(f"Object tested:        {test_object}")
    print(f"Light direction error: {light_errors.mean():.2f}°")
    print(f"Normal map error:      {mean_normal_err:.2f}°")
    print('=' * 50)

    print("\nDisplaying predicted normal map...")
    rps_pred.disp_normalmap()

    return {
        'object': test_object,
        'light_error': light_errors.mean(),
        'normal_error': mean_normal_err
    }

def quick_test_on_training_data(test_object='buddhaPNG'):
    """
    Quick test: Use predictor on one of the training objects to verify pipeline.
    Compare predicted lights vs actual lights.

    Args:
        test_object: Name of object folder to test (e.g., 'buddhaPNG', 'catPNG')
    """
    print("\n" + "=" * 60)
    print(f"QUICK TEST: Verify Pipeline on {test_object}")
    print("=" * 60)

    # Paths relative to project root
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    TRAINING_FOLDER = os.path.join(os.path.dirname(__file__), 'data/training/')

    # Model could be in script dir or project root
    MODEL_PATH_LOCAL = os.path.join(os.path.dirname(__file__), 'light_predictor_v2.pkl')
    MODEL_PATH_ROOT = os.path.join(BASE_DIR, 'light_predictor_v2.pkl')

    # Check which one exists
    if os.path.exists(MODEL_PATH_LOCAL):
        MODEL_PATH = MODEL_PATH_LOCAL
    elif os.path.exists(MODEL_PATH_ROOT):
        MODEL_PATH = MODEL_PATH_ROOT
    else:
        print(f"Model not found at:")
        print(f"  - {MODEL_PATH_LOCAL}")
        print(f"  - {MODEL_PATH_ROOT}")
        print("Please run light_direction_predictor.py first to train the model.")
        return

    predictor = LightDirectionPredictor()
    predictor.load(MODEL_PATH)

    # Test on specified object
    test_folder = os.path.join(TRAINING_FOLDER, test_object)

    if not os.path.exists(test_folder):
        print(f"Test folder not found: {test_folder}")
        return

    gt_lights_path = os.path.join(test_folder, 'light_directions.txt')
    pred_lights_path = os.path.join(test_folder, 'predicted_light_directions.txt')

    # Predict
    predicted_lights = predictor.predict_from_folder(test_folder, pred_lights_path)
    #predicted_lights = predictor.predict_from_folder(test_folder, gt_lights_path)

    # Compare with ground truth
    gt_lights = np.loadtxt(gt_lights_path)

    # Angular error between predicted and GT light directions
    cos_sim = np.sum(gt_lights * predicted_lights, axis=1)
    cos_sim = np.clip(cos_sim, -1, 1)
    light_errors = np.arccos(cos_sim) * 180 / np.pi

    print(f"\nLight Direction Prediction Errors for {test_object}:")
    print(f"  Mean: {light_errors.mean():.2f}°")
    print(f"  Std:  {light_errors.std():.2f}°")
    print(f"  Min:  {light_errors.min():.2f}°")
    print(f"  Max:  {light_errors.max():.2f}°")

    # Now run RPS with predicted lights and compare normals
    print("\n--- Running RPS with Predicted Lights ---")
    mask_path = os.path.join(test_folder, '/metadata/mask.png')

    rps_pred = run_rps_pipeline(
        images_folder=test_folder + '/',
        light_txt_path=pred_lights_path,
        mask_path=mask_path if os.path.exists(mask_path) else None,
        output_normal_path=os.path.join(BASE_DIR, 'test_normal_predicted'),
        method=RPS.L2_SOLVER
    )

    print("\n--- Running RPS with Ground Truth Lights ---")
    rps_gt = run_rps_pipeline(
        images_folder=test_folder + '/',
        light_txt_path=gt_lights_path,
        mask_path=mask_path if os.path.exists(mask_path) else None,
        output_normal_path=os.path.join(BASE_DIR, 'test_normal_gt'),
        method=RPS.L2_SOLVER
    )

    # Compare normal maps
    normal_cos_sim = np.sum(rps_gt.N * rps_pred.N, axis=1)
    normal_cos_sim = np.clip(normal_cos_sim, -1, 1)
    normal_errors = np.arccos(normal_cos_sim) * 180 / np.pi

    if rps_pred.background_ind is not None:
        normal_errors[rps_pred.background_ind] = 0
        foreground_mask = np.ones(len(normal_errors), dtype=bool)
        foreground_mask[rps_pred.background_ind] = False
        mean_normal_err = normal_errors[foreground_mask].mean()
    else:
        mean_normal_err = normal_errors.mean()

    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print('=' * 50)
    print(f"Object tested:        {test_object}")
    print(f"Light direction error: {light_errors.mean():.2f}°")
    print(f"Normal map error:      {mean_normal_err:.2f}°")
    print('=' * 50)

    print("\nDisplaying predicted normal map...")
    rps_pred.disp_normalmap()

    return {
        'object': test_object,
        'light_error': light_errors.mean(),
        'normal_error': mean_normal_err
    }


def test_all_objects():
    """
    Test on all training objects and show comprehensive results.
    """
    print("=" * 60)
    print("COMPREHENSIVE TEST ON ALL OBJECTS")
    print("=" * 60)

    objects = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG',
               'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']

    results = []
    for obj in objects:
        result = quick_test_on_training_data(obj)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Object':<15} {'Light Err':>12} {'Normal Err':>12}")
    print("-" * 40)

    for r in results:
        print(f"{r['object']:<15} {r['light_error']:>10.2f}° {r['normal_error']:>10.2f}°")

    if results:
        avg_light = np.mean([r['light_error'] for r in results])
        avg_normal = np.mean([r['normal_error'] for r in results])

        print("-" * 40)
        print(f"{'AVERAGE':<15} {avg_light:>10.2f}° {avg_normal:>10.2f}°")

        print("\n" + "=" * 60)
        print("QUALITY ASSESSMENT")
        print("=" * 60)
        if avg_normal < 10:
            print("EXCELLENT: Normal maps are highly accurate")
        elif avg_normal < 15:
            print("GOOD: Normal maps are usable with minor artifacts")
        elif avg_normal < 25:
            print("FAIR: Normal maps show visible errors but structure preserved")
        else:
            print("POOR: Normal maps have significant errors")

    return results


if __name__ == '__main__':
    # Choose which to run:

    # Option 1: Quick test on single object
    # quick_test_on_training_data('buddhaPNG')
    quick_test_on_real_images('dimooPNG')

    # Option 2: Test all objects (comprehensive)
    # test_all_objects()

    # Option 3: Full pipeline with your own images
    # full_pipeline_with_training()