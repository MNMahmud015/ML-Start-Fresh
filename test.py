import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 60)
    logger.info("STARTING ML PIPELINE TEST")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("📊 Step 1: Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    logger.info(f"   ✓ Dataset loaded successfully")
    logger.info(f"   ✓ Total samples: {X.shape[0]}")
    logger.info(f"   ✓ Features: {X.shape[1]}")
    logger.info(f"   ✓ Classes: {np.unique(y)} ({len(np.unique(y))} classes)")
    logger.info(f"   ✓ Feature names: {iris.feature_names}")
    
    # Create DataFrame
    logger.info("📋 Step 2: Creating pandas DataFrame...")
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    logger.info(f"   ✓ DataFrame shape: {df.shape}")
    logger.info(f"   ✓ DataFrame info:\n{df.describe().to_string()}")
    
    # Split data
    logger.info("✂️  Step 3: Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logger.info(f"   ✓ Training samples: {X_train.shape[0]}")
    logger.info(f"   ✓ Testing samples: {X_test.shape[0]}")
    logger.info(f"   ✓ Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.2f}")
    
    # Train model
    logger.info("🤖 Step 4: Training RandomForest model (100 estimators)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    logger.info(f"   ✓ Model training completed!")
    logger.info(f"   ✓ Feature importance: {dict(zip(iris.feature_names, model.feature_importances_.round(4)))}")
    
    # Evaluate on train set
    logger.info("📈 Step 5: Evaluating model on training set...")
    train_accuracy = model.score(X_train, y_train)
    logger.info(f"   ✓ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Evaluate on test set
    logger.info("📊 Step 6: Evaluating model on test set...")
    y_pred = model.predict(X_test)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"   ✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"   ✓ Classification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
    
    # Create visualizations
    logger.info("🎨 Step 7: Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ML Pipeline Test - Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot
    logger.info("   ✓ Creating scatter plot...")
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, alpha=0.7, edgecolors='k')
    ax1.set_xlabel('Sepal Length (cm)', fontweight='bold')
    ax1.set_ylabel('Sepal Width (cm)', fontweight='bold')
    ax1.set_title('Feature Space Visualization')
    plt.colorbar(scatter, ax=ax1, label='Class')
    
    # Plot 2: Feature importance
    logger.info("   ✓ Creating feature importance plot...")
    ax2 = axes[0, 1]
    importance = model.feature_importances_
    ax2.barh(iris.feature_names, importance, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Importance', fontweight='bold')
    ax2.set_title('Feature Importance')
    ax2.invert_yaxis()
    
    # Plot 3: Confusion matrix
    logger.info("   ✓ Creating confusion matrix...")
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax3, 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Confusion Matrix')
    
    # Plot 4: Model metrics
    logger.info("   ✓ Creating metrics comparison...")
    ax4 = axes[1, 1]
    metrics = ['Train Accuracy', 'Test Accuracy']
    values = [train_accuracy, test_accuracy]
    colors = ['#2ecc71', '#3498db']
    bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylim([0.9, 1.0])
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Model Performance Metrics')
    ax4.axhline(y=0.95, color='r', linestyle='--', label='95% threshold', linewidth=2)
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("   ✓ Visualizations saved as 'iris_analysis.png'")
    
    logger.info("=" * 60)
    logger.info("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Summary: {len(np.unique(y))} classes | {X.shape[0]} samples | {test_accuracy*100:.2f}% accuracy")
    
except Exception as e:
    logger.error(f"❌ An error occurred: {str(e)}", exc_info=True)
