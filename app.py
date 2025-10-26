#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:33:36 2025

@author: syed
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import gaussian_kde

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(page_title="Supply Chain Analytics Dashboard", layout="wide", page_icon="📦")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# DATA LOADING FUNCTION
# ===========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Data cleaning
    df = df.loc[:, ~df.columns.duplicated()]
    df.rename(columns={
        'Costs': 'Transportation_costs',
        'Lead times': 'Supplier_lead_time'
    }, inplace=True)
    
    df = df.fillna({
        'Shipping carriers': 'Unknown',
        'Customer demographics': 'Unknown'
    })
    
    df['Price'] = df['Price'].astype(float)
    df['Revenue generated'] = df['Revenue generated'].astype(float)
    df['Shipping costs'] = df['Shipping costs'].astype(float)
    
    # Feature engineering
    low_price = df['Price'].quantile(0.25)
    high_price = df['Price'].quantile(0.75)
    bins_price = [0, low_price, high_price, df['Price'].max()]
    labels_price = ['Low cost', 'Medium range', 'Premium']
    df['PriceDistribution'] = pd.cut(df['Price'], bins_price, labels_price)
    
    low_revenue = df['Revenue generated'].quantile(0.25)
    high_revenue = df['Revenue generated'].quantile(0.75)
    bins_revenue = [0, low_revenue, high_revenue, df['Revenue generated'].max()]
    labels_revenue = ['Low Value Cluster', 'Medium Value Cluster', 'High Value Cluster']
    df['Customer_cluster_revenue'] = pd.cut(df['Revenue generated'], bins=bins_revenue, labels=labels_revenue, include_lowest=True)
    
    low_defect = df['Defect rates'].quantile(0.25)
    high_defect = df['Defect rates'].quantile(0.75)
    bins_defect = [0, low_defect, high_defect, df['Defect rates'].max()]
    labels_defect = ['Low Defects', 'Medium Defects', 'High Defects']
    df['Defect_rate_cluster'] = pd.cut(df['Defect rates'], bins=bins_defect, labels=labels_defect, include_lowest=True)
    
    return df

# ===========================
# PAGE FUNCTIONS
# ===========================

def home_page():
    """Empty home page"""
    st.markdown('<p class="main-header">📦 Supply Chain Management Analytics Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Supply Chain Analytics Dashboard
    
    This interactive dashboard provides comprehensive analysis of supply chain data including:
    
    - 📊 **Data Overview**: Explore your dataset statistics and correlations
    - 🛍️ **Product Analysis**: Analyze product performance and pricing
    - 👥 **Customer Segmentation**: Understand customer demographics and behavior
    - 🏭 **Supplier Analysis**: Evaluate supplier performance and reliability
    - 🚚 **Logistics Analysis**: Optimize shipping and transportation
    - 🔍 **Diagnostic Analytics**: Root cause analysis and insights
    
    ---
    
    **To get started:**
    1. Upload your CSV file using the sidebar (or use the default dataset)
    2. Navigate to different sections using the sidebar menu
    3. Explore interactive visualizations and insights
    
    """)
    
    if 'df' in st.session_state and st.session_state.df is not None:
        st.success("✅ Data loaded successfully! Navigate to other pages to start exploring.")


def data_overview_page(df):
    """Data Overview page"""
    st.markdown('<p class="section-header">Data Overview</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📊 Statistics", "🔍 Missing Values", "📈 Correlations"])
    
    with tab1:
        st.subheader("Dataset Preview")
        show_rows = st.slider("Number of rows to display", 10, 100, 50)
        st.dataframe(df.head(show_rows), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        st.download_button(
            label="📥 Download Full Dataset",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='supply_chain_cleaned.csv',
            mime='text/csv',
        )
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab3:
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing[missing > 0])
    
    with tab4:
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
        st.pyplot(fig)


def product_analysis_page(df):
    """Product Analysis page"""
    st.markdown('<p class="section-header">Product Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Top Products", "Price Distribution", "Product Segmentation"])
    
    with tab1:
        st.subheader("Top 10 SKUs by Revenue")
        top_products = df.groupby('SKU').agg({
            'Number of products sold': 'sum',
            'Revenue generated': 'sum'
        }).sort_values(by='Revenue generated', ascending=False).head(10)
        
        st.dataframe(top_products, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_products.index, y=top_products['Revenue generated'], palette='viridis', ax=ax)
        ax.set_title("Top 10 SKUs by Revenue", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Price Distribution Across Products")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Price'], bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_title("Price Distribution Across Products", fontsize=14)
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        st.subheader("Finding Premium Outliers")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df['Price'], color='lightcoral', ax=ax)
        ax.set_title('Finding Premium Outliers Range')
        ax.set_xlabel('Price')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Product Segmentation by Price")
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x=df['PriceDistribution'], palette='viridis', ax=ax)
        ax.set_title('Product Segmentation by Price')
        st.pyplot(fig)
        
        price_stats = df.groupby('PriceDistribution')['Revenue generated'].agg(['count', 'sum', 'mean']).reset_index()
        price_stats.columns = ['Price Segment', 'Count', 'Total Revenue', 'Avg Revenue']
        st.dataframe(price_stats, use_container_width=True)


def customer_segmentation_page(df):
    """Customer Segmentation page"""
    st.markdown('<p class="section-header">Customer Segmentation Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Demographic Analysis", "Customer Clusters"])
    
    with tab1:
        st.subheader("Revenue Distribution by Demographics")
        
        demo_rev_cont = (df.groupby('Customer demographics')['Revenue generated']
                         .sum().reset_index()
                         .sort_values(by='Revenue generated', ascending=False))
        
        st.dataframe(demo_rev_cont, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=demo_rev_cont['Customer demographics'], 
                   y=demo_rev_cont['Revenue generated'], 
                   data=demo_rev_cont, palette='Set2', ax=ax)
        ax.set_title('Revenue Distribution By Demographic Data')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Revenue-based Customer Segmentation")
        
        cluster_counts = df['Customer_cluster_revenue'].value_counts()
        st.write("**Cluster Distribution:**")
        st.write(cluster_counts)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.countplot(x=df['Customer_cluster_revenue'], palette='viridis', ax=ax)
        ax.set_title('Revenue-based Customer Segmentation', fontsize=14)
        ax.set_xlabel("Customer Cluster (by Revenue)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
        
        cal_cluster_revenue = df.groupby('Customer_cluster_revenue')['Revenue generated'].sum().reset_index()
        st.dataframe(cal_cluster_revenue, use_container_width=True)


def supplier_analysis_page(df):
    """Supplier Analysis page"""
    st.markdown('<p class="section-header">Supplier Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Lead Time Analysis", "Cost vs Defect Analysis"])
    
    with tab1:
        st.subheader("Average Lead Time per Supplier")
        
        average_lead_time = df.groupby('Supplier name')['Lead time'].mean().reset_index()
        st.dataframe(average_lead_time, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Supplier name', y='Lead time', data=average_lead_time, palette='viridis', ax=ax)
        ax.set_title('Average Lead Time per Supplier', fontsize=14)
        ax.set_xlabel('Supplier Name')
        ax.set_ylabel('Average Lead Time (days)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Cost vs Defect Rates Trade-off")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Manufacturing costs', y='Defect rates', data=df, hue='Supplier name', s=100, ax=ax)
        ax.set_title("Cost vs Defect Rates")
        st.pyplot(fig)
        
        st.subheader("Manufacturing Cost by Defect Cluster")
        
        manufacturing_cost_defect_total = df.groupby('Defect_rate_cluster')['Manufacturing costs'].sum().reset_index()
        manufacturing_cost_defect_efficiency = df.groupby('Defect_rate_cluster')['Manufacturing costs'].mean().reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.barplot(x='Defect_rate_cluster', y='Manufacturing costs', data=manufacturing_cost_defect_total, ax=axes[0], palette="viridis")
        axes[0].set_title("Total Manufacturing Cost by Defect Cluster")
        
        sns.barplot(x='Defect_rate_cluster', y='Manufacturing costs', data=manufacturing_cost_defect_efficiency, ax=axes[1], palette="magma")
        axes[1].set_title("Average Manufacturing Cost (Efficiency) by Defect Cluster")
        
        st.pyplot(fig)


def logistics_analysis_page(df):
    """Logistics Analysis page"""
    st.markdown('<p class="section-header">Logistics Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Shipping Analysis", "Transportation Modes", "Route Efficiency"])
    
    with tab1:
        st.subheader("Average Shipping Times & Costs by Carrier")
        
        shipping_stats = df.groupby('Shipping carriers').agg({
            'Shipping times': 'mean',
            'Shipping costs': 'mean'
        }).reset_index()
        
        st.dataframe(shipping_stats, use_container_width=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.barplot(x='Shipping carriers', y='Shipping times', data=shipping_stats, color='skyblue', ax=axes[0])
        axes[0].set_title("Avg. Shipping Time by Carrier")
        axes[0].set_ylabel("Days")
        
        sns.barplot(x='Shipping carriers', y='Shipping costs', data=shipping_stats, color='red', ax=axes[1])
        axes[1].set_title("Avg. Shipping Cost by Carrier")
        axes[1].set_ylabel("Cost")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Most Used Transportation Modes")
        
        transport_counts = df['Transportation modes'].value_counts().reset_index()
        transport_counts.columns = ['Mode', 'Count']
        st.dataframe(transport_counts, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x='Transportation modes', data=df, palette='Set3', ax=ax)
        ax.set_xlabel('Transport Mode')
        ax.set_ylabel('Transport Frequency')
        ax.set_title('Most Used Transportation')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Route-wise Cost Efficiency")
        
        route_wise_cost = df.groupby('Routes').agg({
            'Transportation_costs': 'mean'
        }).reset_index()
        
        st.dataframe(route_wise_cost, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Routes', y='Transportation_costs', data=route_wise_cost, palette='coolwarm', ax=ax)
        ax.set_title("Route-wise Cost Efficiency")
        ax.set_xlabel('Routes')
        ax.set_ylabel('Cost Efficiency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)


def diagnostic_analytics_page(df):
    """Diagnostic Analytics page"""
    st.markdown('<p class="section-header">Diagnostic Analytics (Root Cause Analysis)</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Inventory Bottlenecks", "Supplier Reliability"])
    
    with tab1:
        st.subheader("Stock Levels vs Lead Time Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Lead time', y='Stock levels', data=df, hue='Supplier name', s=100, ax=ax)
        ax.set_xlabel('Supplier Lead Time (days)')
        ax.set_ylabel('Stock Level')
        ax.set_title('Stock Level vs Lead Time')
        ax.axvline(x=df['Lead time'].median(), color='red', linestyle='--', label='Median Lead Time')
        ax.axhline(y=df['Stock levels'].median(), color='blue', linestyle='--', label='Median Stock Level')
        ax.legend()
        st.pyplot(fig)
        
        correlation = df['Lead time'].corr(df['Stock levels'])
        st.metric("Correlation (Lead Time vs Stock Levels)", f"{correlation:.4f}")
    
    with tab2:
        st.subheader("Lead Time vs Defect Rates by Supplier")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Lead time', y='Defect rates', data=df, hue='Supplier name', s=100, ax=ax)
        ax.set_xlabel('Lead Time (Days)')
        ax.set_ylabel('Defect Rate')
        ax.set_title('Lead Time vs Defect Rate by Suppliers')
        ax.axvline(x=df['Lead time'].median(), color='red', linestyle='--')
        ax.axhline(y=df['Defect rates'].median(), color='blue', linestyle='--')
        st.pyplot(fig)
        
        correlation = df['Lead time'].corr(df['Defect rates'])
        st.metric("Correlation (Lead Time vs Defect Rates)", f"{correlation:.4f}")



def main_analytics(df):
   
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Revenue", f"${df['Revenue generated'].sum():,.2f}")
    with col3:
        st.metric("Total Products Sold", f"{df['Number of products sold'].sum():,}")
    with col4:
        st.metric("Unique SKUs", f"{df['SKU'].nunique():,}")
    
    st.markdown("---")
    
    # Show full data preview
    with st.expander("📋 View Full Dataset", expanded=False):
        st.dataframe(df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue by Product Type")
        revenue_by_type = df.groupby('Product type')['Revenue generated'].sum().reset_index()
        fig = px.pie(revenue_by_type, values='Revenue generated', names='Product type', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚚 Shipping Carriers Distribution")
        carrier_dist = df['Shipping carriers'].value_counts().reset_index()
        carrier_dist.columns = ['Carrier', 'Count']
        fig = px.bar(carrier_dist, x='Carrier', y='Count', color='Carrier')
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# MAIN APP LOGIC
# ===========================

def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Sidebar
    st.sidebar.title("📁 Data Upload")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Supply Chain Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.sidebar.success("✅ File loaded successfully!")
    else:
        try:
            st.session_state.df = load_data('supply_chain_data.csv')
            st.sidebar.info("ℹ️ Using default dataset")
        except:
            st.session_state.df = None
    
    # Show data preview in sidebar if data is loaded
    if st.session_state.df is not None:
        with st.sidebar.expander("📊 Data Preview", expanded=False):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            st.write(f"**Total rows:** {len(st.session_state.df):,}")
            st.write(f"**Total columns:** {len(st.session_state.df.columns)}")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "🔥 Executive Summary",
        "📊 Data Overview",
        "🛍️ Product Analysis",
        "👥 Customer Segmentation",
        "🏭 Supplier Analysis",
        "🚚 Logistics Analysis",
        "🔍 Diagnostic Analytics"
    ])
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    
    elif st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file to begin analysis.")
    elif page == "🔥 Executive Summary":
        main_analytics(st.session_state.df)
    elif page == "📊 Data Overview":
        data_overview_page(st.session_state.df)
    
    elif page == "🛍️ Product Analysis":
        product_analysis_page(st.session_state.df)
    
    elif page == "👥 Customer Segmentation":
        customer_segmentation_page(st.session_state.df)
    
    elif page == "🏭 Supplier Analysis":
        supplier_analysis_page(st.session_state.df)
    
    elif page == "🚚 Logistics Analysis":
        logistics_analysis_page(st.session_state.df)
    
    elif page == "🔍 Diagnostic Analytics":
        diagnostic_analytics_page(st.session_state.df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            📊 Supply Chain Analytics Dashboard | Built with Streamlit | Advance Statistics MBS © 2025  | Created By: Dr. Mehtab Alam SYED
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


