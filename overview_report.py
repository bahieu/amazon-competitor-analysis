import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from ai_insights import AIInsights
import numpy as np

def has_significant_change(data, threshold=0.1):
    if len(data) < 2:
        return False
    cv = data.std() / data.mean() if data.mean() != 0 else 0
    return cv > threshold

def create_overview_report(unique_df):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='VietnameseNormal', fontSize=10, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseHeading2', fontSize=14, leading=16, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseHeading3', fontSize=12, leading=14, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseTitle', fontSize=16, leading=18, spaceAfter=12))

    story = []
    ai = AIInsights()

    story.append(Paragraph("BÁO CÁO TỔNG QUAN THỊ TRƯỜNG", styles['VietnameseTitle']))
    story.append(Spacer(1, 12))

    # Tổng quan số liệu
    total_revenue = unique_df['revenue'].sum()
    total_sales = unique_df['bought in past month'].sum()
    avg_price = unique_df['price $'].median()
    story.append(Paragraph(f"Tổng doanh thu: ${total_revenue:,.2f}", styles['VietnameseNormal']))
    story.append(Paragraph(f"Tổng số lượng bán trong tháng qua: {total_sales:,.0f} sản phẩm", styles['VietnameseNormal']))
    story.append(Paragraph(f"Giá trung bình trên thị trường: ${avg_price:.2f}", styles['VietnameseNormal']))
    story.append(Spacer(1, 12))

    # Top categories
    cat_stats = unique_df.groupby('category')['bought in past month'].sum().sort_values(ascending=False)
    cat_stats_percent = (cat_stats / cat_stats.sum() * 100).head(5).reset_index()
    if not cat_stats_percent.empty:
        fig = px.bar(
            cat_stats_percent,
            x='category',
            y='bought in past month',
            title="Top 5 Categories by Percentage of Sales",
            labels={'category': 'Category', 'bought in past month': 'Percentage (%)'},
            text_auto='.1f'
        )
        fig.update_layout(xaxis_tickangle=45)
        story.append(fig)
    story.append(Spacer(1, 12))

    # Top subcategories
    sub_stats = unique_df.groupby('sub_category')['bought in past month'].sum().sort_values(ascending=False).head(10)
    top_subcategories = sub_stats.index.tolist()
    if not sub_stats.empty:
        sub_stats_df = sub_stats.reset_index()
        fig = px.bar(
            sub_stats_df,
            x='sub_category',
            y='bought in past month',
            title="Top 10 Sub-Categories by Bought in Past Month",
            labels={'sub_category': 'Sub-Category', 'bought in past month': 'Bought in Past Month'},
            text_auto='d'
        )
        fig.update_layout(xaxis_tickangle=45)
        story.append(fig)
    story.append(Spacer(1, 12))

    # Market trends over time
    if 'date' in unique_df.columns:
        market_data = unique_df.groupby('date').agg({
            'price $': 'median',
            'bought in past month': 'sum',
            'revenue': 'sum'
        }).reset_index()
        market_data['date'] = pd.to_datetime(market_data['date'])
        market_data = market_data.dropna()

        significant_change = any([
            has_significant_change(market_data['price $']),
            has_significant_change(market_data['bought in past month']),
            has_significant_change(market_data['revenue'])
        ])

        if significant_change and not market_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=market_data['date'],
                y=market_data['price $'],
                mode='lines+markers',
                name='Median Price ($)',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=market_data['date'],
                y=market_data['bought in past month'],
                mode='lines+markers',
                name='Bought in Past Month',
                yaxis='y2',
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=market_data['date'],
                y=market_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                yaxis='y3',
                line=dict(color='green')
            ))
            fig.update_layout(
                title="Market Trends Over Time",
                xaxis=dict(title='Date'),
                yaxis=dict(title='Median Price ($)', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                yaxis2=dict(title='Bought in Past Month', titlefont=dict(color='orange'), tickfont=dict(color='orange'), overlaying='y', side='right'),
                yaxis3=dict(title='Revenue', titlefont=dict(color='green'), tickfont=dict(color='green'), overlaying='y', side='right', anchor='free', position=0.95),
                xaxis_tickangle=45
            )
            story.append(fig)
        else:
            story.append(Paragraph("Không vẽ biểu đồ xu hướng thị trường vì dữ liệu không có sự thay đổi đáng kể hoặc không có dữ liệu.", styles['VietnameseNormal']))
        story.append(Spacer(1, 12))

    # Generate insights (đã loại bỏ detect_anomalies và time_series_analysis)
    insights = ai.generate_insights(unique_df)
    filtered_insights = [
        insight for insight in insights
        if not (
            insight.startswith("• Cảnh báo bất thường") or
            "xu hướng tăng" in insight or
            "xu hướng giảm" in insight
        )
    ]
    if filtered_insights:
        story.append(Paragraph("Nhận Xét Tổng Quan:", styles['VietnameseHeading3']))
        for insight in filtered_insights:
            story.append(Paragraph(f"• {insight}", styles['VietnameseNormal']))

    return story, top_subcategories

def main(unique_df):
    return create_overview_report(unique_df)