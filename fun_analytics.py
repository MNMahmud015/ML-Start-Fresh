import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import seaborn as sns
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 70)
    logger.info("🎬 FUN DATA ANALYSIS & VISUALIZATION PLAYGROUND")
    logger.info("=" * 70)
    
    # ============ PART 1: Generate Fun Synthetic Data ============
    logger.info("📊 Part 1: Generating synthetic sales data...")
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n_products = 5
    products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Smartwatch']
    
    data = {
        'Date': [],
        'Product': [],
        'Sales': [],
        'Revenue': [],
        'Customer_Rating': []
    }
    
    for date in dates:
        for product in products:
            data['Date'].append(date)
            data['Product'].append(product)
            # Realistic sales with trends
            base_sales = 10 + 5 * np.sin(date.dayofyear / 365 * 2 * np.pi)
            noise = np.random.normal(0, 3)
            sales = max(0, int(base_sales + noise))
            data['Sales'].append(sales)
            data['Revenue'].append(sales * np.random.uniform(500, 2000))
            data['Customer_Rating'].append(np.random.uniform(3.5, 5.0))
    
    df = pd.DataFrame(data)
    logger.info(f"   ✓ Generated {len(df)} sales records across {n_products} products")
    logger.info(f"   ✓ Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    logger.info(f"   ✓ Total revenue: ${df['Revenue'].sum():,.2f}")
    logger.info(f"   ✓ Average rating: {df['Customer_Rating'].mean():.2f}/5.0")
    
    # ============ PART 2: Data Analysis ============
    logger.info("🔍 Part 2: Analyzing sales data...")
    
    product_stats = df.groupby('Product').agg({
        'Sales': ['sum', 'mean', 'std'],
        'Revenue': 'sum',
        'Customer_Rating': 'mean'
    }).round(2)
    
    logger.info("   ✓ Product performance summary:")
    for product in products:
        product_data = df[df['Product'] == product]
        total_sales = product_data['Sales'].sum()
        avg_rating = product_data['Customer_Rating'].mean()
        total_revenue = product_data['Revenue'].sum()
        logger.info(f"     • {product:15s} | Sales: {total_sales:4d} | Revenue: ${total_revenue:,.0f} | Rating: {avg_rating:.2f}⭐")
    
    # ============ PART 3: Monthly Trends ============
    logger.info("📈 Part 3: Analyzing monthly trends...")
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Sales'].sum()
    monthly_revenue = df.groupby('Month')['Revenue'].sum()
    
    logger.info(f"   ✓ Best month: {monthly_revenue.idxmax()} (${monthly_revenue.max():,.2f})")
    logger.info(f"   ✓ Worst month: {monthly_revenue.idxmin()} (${monthly_revenue.min():,.2f})")
    logger.info(f"   ✓ Average monthly revenue: ${monthly_revenue.mean():,.2f}")
    
    # ============ PART 4: Create Visualizations ============
    logger.info("🎨 Part 4: Creating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('🎯 Sales Analytics Dashboard - 2024', fontsize=20, fontweight='bold', y=0.995)
    
    # Subplot 1: Daily sales trend
    logger.info("   ✓ Creating daily sales trend...")
    ax1 = plt.subplot(3, 3, 1)
    daily_sales = df.groupby('Date')['Sales'].sum()
    ax1.fill_between(daily_sales.index, daily_sales.values, alpha=0.3, color='steelblue')
    ax1.plot(daily_sales.index, daily_sales.values, linewidth=2, color='steelblue')
    ax1.set_title('Daily Sales Trend', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Units Sold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Subplot 2: Product sales comparison
    logger.info("   ✓ Creating product comparison...")
    ax2 = plt.subplot(3, 3, 2)
    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=True)
    colors = cm.Spectral(np.linspace(0, 1, len(product_sales)))
    bars = ax2.barh(product_sales.index, product_sales.values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_title('Total Sales by Product', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Units Sold')
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{int(bar.get_width())}', ha='left', va='center', fontweight='bold')
    
    # Subplot 3: Revenue distribution
    logger.info("   ✓ Creating revenue distribution...")
    ax3 = plt.subplot(3, 3, 3)
    product_revenue = df.groupby('Product')['Revenue'].sum()
    wedges, texts, autotexts = ax3.pie(product_revenue.values, labels=product_revenue.index, 
                                        autopct='%1.1f%%', startangle=90, colors=colors,
                                        explode=[0.05]*len(products))
    ax3.set_title('Revenue Distribution', fontweight='bold', fontsize=11)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Subplot 4: Monthly revenue
    logger.info("   ✓ Creating monthly revenue...")
    ax4 = plt.subplot(3, 3, 4)
    monthly_rev = df.groupby('Month')['Revenue'].sum()
    ax4.bar(range(len(monthly_rev)), monthly_rev.values, color='darkgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_title('Monthly Revenue', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Revenue ($)')
    ax4.set_xlabel('Month')
    ax4.set_xticks(range(len(monthly_rev)))
    ax4.set_xticklabels([str(m) for m in monthly_rev.index], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: Customer ratings by product
    logger.info("   ✓ Creating rating analysis...")
    ax5 = plt.subplot(3, 3, 5)
    rating_data = [df[df['Product'] == p]['Customer_Rating'].values for p in products]
    bp = ax5.boxplot(rating_data, labels=products, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax5.set_title('Customer Ratings Distribution', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Rating (1-5)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Subplot 6: Sales heatmap by product and month
    logger.info("   ✓ Creating heatmap...")
    ax6 = plt.subplot(3, 3, 6)
    pivot_data = df.pivot_table(values='Sales', index='Product', columns='Month', aggfunc='sum')
    sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Units Sold'})
    ax6.set_title('Sales Heatmap (Product vs Month)', fontweight='bold', fontsize=11)
    ax6.set_xlabel('')
    
    # Subplot 7: Scatter plot - Sales vs Revenue
    logger.info("   ✓ Creating scatter analysis...")
    ax7 = plt.subplot(3, 3, 7)
    for i, product in enumerate(products):
        prod_data = df[df['Product'] == product]
        ax7.scatter(prod_data['Sales'], prod_data['Revenue'], 
                   label=product, alpha=0.6, s=50, color=colors[i])
    ax7.set_title('Sales vs Revenue Correlation', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Units Sold')
    ax7.set_ylabel('Revenue ($)')
    ax7.legend(fontsize=8, loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # Subplot 8: Top 10 sales days
    logger.info("   ✓ Creating top sales days...")
    ax8 = plt.subplot(3, 3, 8)
    top_days = df.groupby('Date')['Sales'].sum().nlargest(10)
    ax8.barh(range(len(top_days)), top_days.values, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax8.set_yticks(range(len(top_days)))
    ax8.set_yticklabels([d.strftime('%m-%d') for d in top_days.index], fontsize=9)
    ax8.set_title('Top 10 Sales Days', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Units Sold')
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Subplot 9: Key Metrics Summary
    logger.info("   ✓ Creating metrics summary...")
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    metrics_text = f"""
    📊 KEY METRICS SUMMARY
    
    Total Sales: {df['Sales'].sum():,} units
    Total Revenue: ${df['Revenue'].sum():,.0f}
    
    Avg Daily Sales: {df.groupby('Date')['Sales'].sum().mean():.1f} units
    Avg Customer Rating: {df['Customer_Rating'].mean():.2f}⭐
    
    Best Product: {df.groupby('Product')['Revenue'].sum().idxmax()}
    Best Month: {monthly_revenue.idxmax()}
    
    Unique Customers: {np.random.randint(5000, 15000):,}
    Days Analyzed: {len(daily_sales)}
    """
    
    ax9.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
    logger.info("   ✓ Dashboard saved as 'sales_dashboard.png'")
    
    # ============ PART 5: Fun Stats ============
    logger.info("🎉 Part 5: Fun Statistics...")
    
    best_day = df.groupby('Date')['Revenue'].sum().idxmax()
    best_day_revenue = df.groupby('Date')['Revenue'].sum().max()
    worst_performer = df.groupby('Product')['Customer_Rating'].mean().idxmin()
    best_performer = df.groupby('Product')['Customer_Rating'].mean().idxmax()
    
    logger.info(f"   🏆 Best selling day: {best_day.strftime('%A, %B %d, %Y')} (${best_day_revenue:,.2f})")
    logger.info(f"   ⭐ Most loved product: {best_performer}")
    logger.info(f"   😞 Needs improvement: {worst_performer}")
    logger.info(f"   💰 Average transaction value: ${df['Revenue'].sum() / len(df):,.2f}")
    
    logger.info("=" * 70)
    logger.info("✅ ANALYSIS COMPLETE! Check 'sales_dashboard.png' for visualizations")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"❌ Error occurred: {str(e)}", exc_info=True)
