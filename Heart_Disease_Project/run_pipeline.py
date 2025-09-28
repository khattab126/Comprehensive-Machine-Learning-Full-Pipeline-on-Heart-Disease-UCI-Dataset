"""
Main Pipeline Execution Script
Runs the complete heart disease prediction pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
            if result.stdout:
                print("📝 Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False
    
    return True

def main():
    """Main pipeline execution"""
    print("🚀 HEART DISEASE PREDICTION PIPELINE")
    print("="*60)
    print("Starting comprehensive ML pipeline execution...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('ui', exist_ok=True)
    
    # Pipeline steps
    pipeline_steps = [
        ("data_preprocessing.py", "Data Preprocessing & Cleaning"),
        ("notebooks/02_pca_analysis.py", "PCA Analysis & Dimensionality Reduction"),
        ("notebooks/03_feature_selection.py", "Feature Selection"),
        ("notebooks/04_supervised_learning.py", "Supervised Learning Models"),
        ("notebooks/05_unsupervised_learning.py", "Unsupervised Learning (Clustering)"),
        ("notebooks/06_hyperparameter_tuning.py", "Hyperparameter Tuning")
    ]
    
    # Execute pipeline steps
    success_count = 0
    total_steps = len(pipeline_steps)
    
    for script, description in pipeline_steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline stopped due to error in {description}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 All pipeline steps completed successfully!")
        print("\n📁 Generated files:")
        print("   - data/heart_disease_processed.csv")
        print("   - data/heart_disease_pca.csv")
        print("   - data/heart_disease_selected_features.csv")
        print("   - models/ (trained models)")
        print("   - results/ (analysis plots and metrics)")
        
        print("\n🚀 To run the Streamlit app:")
        print("   streamlit run ui/app.py")
        
        print("\n📊 To view results:")
        print("   Check the 'results/' directory for generated visualizations")
        
    else:
        print("❌ Pipeline execution incomplete. Please check errors above.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
