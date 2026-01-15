import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 70)
    logger.info("🤖 FUN ML PROJECT: DEVELOPER PRODUCTIVITY PREDICTOR")
    logger.info("=" * 70)
    
    # ============ PART 1: Generate Fun Developer Data ============
    logger.info("📊 Part 1: Generating synthetic developer data...")
    np.random.seed(42)
    
    n_developers = 200
    
    data = {
        'Coffee_Cups_Per_Day': np.random.randint(1, 10, n_developers),
        'Stack_Overflow_Visits': np.random.randint(1, 50, n_developers),
        'GitHub_Commits': np.random.randint(0, 100, n_developers),
        'Slack_Messages': np.random.randint(0, 500, n_developers),
        'Lines_of_Code': np.random.randint(100, 5000, n_developers),
        'Debug_Sessions': np.random.randint(0, 30, n_developers),
        'Hours_Slept': np.random.uniform(4, 8, n_developers),
        'Bugs_Fixed': np.random.randint(0, 50, n_developers),
        'Code_Reviews_Done': np.random.randint(0, 20, n_developers),
    }
    
    df = pd.DataFrame(data)
    
    # Create target: Productivity Score (0-100)
    df['Productivity_Score'] = (
        df['Coffee_Cups_Per_Day'] * 2 +
        df['GitHub_Commits'] * 0.6 +
        df['Bugs_Fixed'] * 1.2 +
        df['Code_Reviews_Done'] * 1.5 +
        df['Hours_Slept'] * 3 -
        df['Slack_Messages'] * 0.1 -
        df['Debug_Sessions'] * 0.8 +
        np.random.normal(0, 15, n_developers)
    )
    df['Productivity_Score'] = np.clip(df['Productivity_Score'], 0, 100)
    
    # Create a "Ninja Developer" label (top 40% productivity)
    df['Is_Ninja_Developer'] = (df['Productivity_Score'] > df['Productivity_Score'].quantile(0.5)).astype(int)
    
    logger.info(f"   ✓ Generated data for {n_developers} developers")
    logger.info(f"   ✓ Average productivity score: {df['Productivity_Score'].mean():.2f}")
    logger.info(f"   ✓ Ninja developers: {df['Is_Ninja_Developer'].sum()} ({df['Is_Ninja_Developer'].mean()*100:.1f}%)")
    
    # ============ PART 2: Data Analysis ============
    logger.info("🔍 Part 2: Analyzing developer patterns...")
    
    logger.info(f"   ✓ Most productive developers drink {df[df['Productivity_Score'] > 70]['Coffee_Cups_Per_Day'].mean():.1f} cups of coffee/day")
    logger.info(f"   ✓ Average commits per ninja: {df[df['Is_Ninja_Developer'] == 1]['GitHub_Commits'].mean():.1f}")
    logger.info(f"   ✓ Correlation between coffee and productivity: {df['Coffee_Cups_Per_Day'].corr(df['Productivity_Score']):.3f}")
    logger.info(f"   ✓ Correlation between sleep and productivity: {df['Hours_Slept'].corr(df['Productivity_Score']):.3f}")
    
    # ============ PART 3: ML Model 1 - Predict Productivity Score ============
    logger.info("🤖 Part 3: Training Model 1 - Productivity Predictor...")
    
    X = df.drop(['Productivity_Score', 'Is_Ninja_Developer'], axis=1)
    y_continuous = df['Productivity_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_continuous, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_regression = RandomForestRegressor(n_estimators=50, random_state=42)
    model_regression.fit(X_train_scaled, y_train)
    
    y_pred_continuous = model_regression.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_continuous)
    rmse = np.sqrt(mse)
    
    logger.info(f"   ✓ Model trained!")
    logger.info(f"   ✓ RMSE: {rmse:.2f}")
    logger.info(f"   ✓ Sample prediction: {y_pred_continuous[0]:.1f} vs actual: {y_test.iloc[0]:.1f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_regression.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    logger.info(f"   ✓ Top productivity factors:")
    for idx, row in feature_importance.head(3).iterrows():
        logger.info(f"     • {row['Feature']}: {row['Importance']:.3f}")
    
    # ============ PART 4: ML Model 2 - Ninja Developer Classifier ============
    logger.info("🥷 Part 4: Training Model 2 - Ninja Developer Classifier...")
    
    y_binary = df['Is_Ninja_Developer']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)
    
    model_classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model_classifier.fit(X_train_c_scaled, y_train_c)
    
    y_pred_binary = model_classifier.predict(X_test_c_scaled)
    accuracy = accuracy_score(y_test_c, y_pred_binary)
    
    logger.info(f"   ✓ Model trained!")
    logger.info(f"   ✓ Accuracy: {accuracy:.2%}")
    logger.info(f"   ✓ Classification Report:")
    for line in classification_report(y_test_c, y_pred_binary, target_names=['Regular Dev', 'Ninja Dev']).split('\n'):
        if line.strip():
            logger.info(f"     {line}")
    
    # ============ PART 5: Create Visualizations ============
    logger.info("🎨 Part 5: Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('🚀 Developer Productivity ML Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    # Plot 1: Feature Importance
    logger.info("   ✓ Creating feature importance...")
    ax1 = plt.subplot(2, 3, 1)
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_sorted)))
    ax1.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Importance'], 
            color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Feature Importance for Productivity', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Importance Score')
    
    # Plot 2: Productivity Distribution
    logger.info("   ✓ Creating productivity distribution...")
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df['Productivity_Score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(df['Productivity_Score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Productivity_Score"].mean():.1f}')
    ax2.set_title('Productivity Score Distribution', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Productivity Score (0-100)')
    ax2.set_ylabel('Number of Developers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coffee vs Productivity
    logger.info("   ✓ Creating coffee analysis...")
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(df['Coffee_Cups_Per_Day'], df['Productivity_Score'], 
                         c=df['Hours_Slept'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    ax3.set_title('Coffee vs Productivity (colored by Sleep)', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Coffee Cups/Day')
    ax3.set_ylabel('Productivity Score')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Hours Slept')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Predictions vs Actual
    logger.info("   ✓ Creating predictions comparison...")
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_test, y_pred_continuous, alpha=0.6, s=80, color='purple', edgecolors='black', linewidth=1)
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_title(f'Model Predictions (RMSE: {rmse:.2f})', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Actual Productivity Score')
    ax4.set_ylabel('Predicted Productivity Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Ninja vs Regular Developer
    logger.info("   ✓ Creating ninja classification...")
    ax5 = plt.subplot(2, 3, 5)
    ninja_stats = df.groupby('Is_Ninja_Developer')[['Coffee_Cups_Per_Day', 'GitHub_Commits', 'Bugs_Fixed']].mean()
    x = np.arange(len(ninja_stats.columns))
    width = 0.35
    ax5.bar(x - width/2, ninja_stats.iloc[0], width, label='Regular Dev', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.bar(x + width/2, ninja_stats.iloc[1], width, label='Ninja Dev', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_title('Ninja vs Regular Developers', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Average Value')
    ax5.set_xticks(x)
    ax5.set_xticklabels(ninja_stats.columns, rotation=15, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Classification Confusion Matrix
    logger.info("   ✓ Creating confusion matrix...")
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_c, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax6,
                xticklabels=['Regular', 'Ninja'], yticklabels=['Regular', 'Ninja'],
                cbar_kws={'label': 'Count'})
    ax6.set_title(f'Classifier Performance (Accuracy: {accuracy:.2%})', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Actual')
    ax6.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('developer_ml_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("   ✓ Visualizations saved as 'developer_ml_analysis.png'")
    
    # ============ PART 6: Fun Predictions ============
    logger.info("🎉 Part 6: Making predictions for test developers...")
    
    sample_developer = X_test_scaled[0:1]
    predicted_score = model_regression.predict(sample_developer)[0]
    is_ninja = model_classifier.predict(sample_developer)[0]
    
    logger.info(f"   Sample Developer Profile:")
    logger.info(f"   ✓ Predicted Productivity Score: {predicted_score:.1f}/100")
    logger.info(f"   ✓ Is Ninja Developer: {'🥷 YES!' if is_ninja else '👨‍💻 Regular Dev'}")
    logger.info(f"   ✓ Confidence: {model_classifier.predict_proba(sample_developer).max():.2%}")
    
    logger.info("=" * 70)
    logger.info("✅ ML ANALYSIS COMPLETE! Check 'developer_ml_analysis.png'")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"❌ Error occurred: {str(e)}", exc_info=True)
