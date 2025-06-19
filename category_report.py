import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import asyncio
from ai_insights import AIInsights
import numpy as np
from utils import download_images, get_price_trend

def has_significant_change(data, threshold=0.1):
    if len(data) < 2:
        return False
    cv = data.std() / data.mean() if data.mean() != 0 else 0
    return cv > threshold

def analyze_category(unique_df, top_subcategories, category_to_analyze, categories_to_analyze):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='VietnameseNormal', fontSize=10, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseHeading2', fontSize=14, leading=16, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseHeading3', fontSize=12, leading=14, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='VietnameseCaption', fontSize=8, leading=10, spaceAfter=4, alignment=1))

    story = []
    ai = AIInsights()

    story.append(Paragraph(f'<a name="{category_to_analyze.replace(" ", "_").replace("&", "and")}"/>', styles['VietnameseNormal']))
    story.append(Paragraph(f"Phân Tích Chi Tiết Category '{category_to_analyze}'", styles['VietnameseHeading2']))
    kd_df = unique_df[unique_df['category'] == category_to_analyze]

    story.append(Paragraph("Top 5 Sản Phẩm Nổi Bật Trong Toàn Category:", styles['VietnameseHeading3']))
    top_5_products = kd_df.nlargest(5, 'bought in past month')[['asin', 'bought in past month', 'revenue', 'price $', 'creation date', 'variation', 'ratings', 'review count', 'seller country/region']]
    if not top_5_products.empty:
        story.append(Paragraph("Trong đó, các sản phẩm nổi bật bao gồm:", styles['VietnameseNormal']))
        for idx, product in top_5_products.iterrows():
            asin = product['asin']
            bought = product['bought in past month']
            revenue = product['revenue']
            price = product['price $']
            brand = kd_df[kd_df['asin'] == asin]['brand'].iloc[0] if 'brand' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['brand'].iloc[0]) else "không xác định"
            creation_date = product['creation date'] if 'creation date' in kd_df.columns and not pd.isna(product['creation date']) else "Không có dữ liệu"
            variation = product['variation'] if 'variation' in kd_df.columns and not pd.isna(product['variation']) else "Không có dữ liệu"
            ratings = product['ratings'] if 'ratings' in kd_df.columns and not pd.isna(product['ratings']) else "Không có dữ liệu"
            review_count = product['review count'] if 'review count' in kd_df.columns and not pd.isna(product['review count']) else "Không có dữ liệu"
            seller_country = product['seller country/region'] if 'seller country/region' in kd_df.columns and not pd.isna(product['seller country/region']) else "Không có dữ liệu"
            comment = (
                f"• ASIN '{asin}' thuộc thương hiệu {brand} có giá ${price:.2f}, bán được {bought:,.0f} sản phẩm, doanh thu ${revenue:,.2f}. "
                f"Ngày ra mắt: {creation_date}, Số lượng biến thể: {variation}, Điểm đánh giá: {ratings}, Số lượng đánh giá: {review_count}, Thuộc seller từ: {seller_country}."
            )
            story.append(Paragraph(comment, styles['VietnameseNormal']))

        image_elements = []
        mentioned_asins_sorted = top_5_products['asin'].tolist()
        image_urls = []
        for asin in mentioned_asins_sorted:
            image_url = kd_df[kd_df['asin'] == asin]['image url'].iloc[0] if 'image url' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['image url'].iloc[0]) else None
            if image_url:
                image_urls.append(image_url)
            else:
                image_urls.append(None)

        images = asyncio.run(download_images(image_urls))

        for idx, asin in enumerate(mentioned_asins_sorted):
            image_buf = images[idx]
            brand = kd_df[kd_df['asin'] == asin]['brand'].iloc[0] if 'brand' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['brand'].iloc[0]) else "không xác định"
            
            asin_data = kd_df[kd_df['asin'] == asin].sort_values(by='date', ascending=False)
            price = asin_data['price $'].iloc[0] if not asin_data.empty else 0.0
            
            url = kd_df[kd_df['asin'] == asin]['url'].iloc[0] if 'url' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['url'].iloc[0]) else f"https://www.amazon.com/dp/{asin}"

            trend = get_price_trend(asin, unique_df)
            trend_symbol = ""
            if trend == "up":
                trend_symbol = '<span style="color:green; font-weight:bold;">↑</span>'
            elif trend == "down":
                trend_symbol = '<span style="color:red; font-weight:bold;">↓</span>'

            caption_text = f"ASIN {asin} - Thương hiệu {brand} - Giá ${price:.2f}"
            caption = (caption_text, url, trend_symbol)

            if image_buf:
                image_elements.append([image_buf, caption])
            else:
                image_elements.append([Paragraph(f"Hình ảnh không khả dụng cho ASIN {asin}", styles['VietnameseNormal']), Spacer(1, 12)])

        if image_elements:
            image_table = []
            row = []
            for i, elem in enumerate(image_elements):
                row.append(elem)
                if (i + 1) % 3 == 0 or (i + 1) == len(image_elements):
                    image_table.append(row)
                    row = []
            if row:
                image_table.append(row)
            story.append(Table(image_table, colWidths=[150] * 3))
            story.append(Spacer(1, 12))

    sub_revenue = kd_df.groupby('sub_category')['revenue'].sum().sort_values(ascending=False).head(10)
    top_10_subcategories = sub_revenue.index.tolist() if not sub_revenue.empty else []
    all_subcategories = kd_df['sub_category'].unique()
    for sub_cat in ['Tumbler', 'Drinking Jars']:
        if sub_cat in all_subcategories and sub_cat not in top_10_subcategories:
            top_10_subcategories.append(sub_cat)

    if not sub_revenue.empty:
        sub_revenue_df = sub_revenue.reset_index()
        fig = px.bar(
            sub_revenue_df,
            x='sub_category',
            y='revenue',
            title=f"Top 10 Sub-Categories by Revenue in {category_to_analyze}",
            labels={'sub_category': 'Sub-Category', 'revenue': 'Revenue ($)'},
            text_auto='.2s'
        )
        fig.update_layout(xaxis_tickangle=45)
        story.append(fig)
    story.append(Spacer(1, 12))

    sub_sales = kd_df.groupby('sub_category')['bought in past month'].sum()
    if top_10_subcategories and not sub_sales.empty:
        sub_sales = sub_sales.loc[sub_sales.index.isin(top_10_subcategories)]
        sub_sales_df = sub_sales.reset_index()
        fig = px.bar(
            sub_sales_df,
            x='sub_category',
            y='bought in past month',
            title=f"Sub-Categories by Bought in Past Month in {category_to_analyze}",
            labels={'sub_category': 'Sub-Category', 'bought in past month': 'Bought in Past Month'},
            text_auto='d'
        )
        fig.update_layout(xaxis_tickangle=45)
        story.append(fig)
    story.append(Spacer(1, 12))

    sub_price_data = kd_df[kd_df['sub_category'].isin(top_10_subcategories)]
    sub_price_data = sub_price_data.dropna(subset=['price $'])
    valid_subcategories = [sub_cat for sub_cat in top_10_subcategories if not sub_price_data[sub_price_data['sub_category'] == sub_cat]['price $'].empty]
    if valid_subcategories:
        sub_price_data = sub_price_data[sub_price_data['sub_category'].isin(valid_subcategories)]
        fig = px.box(
            sub_price_data,
            x='sub_category',
            y='price $',
            title=f"Price Distribution of Sub-Categories in {category_to_analyze}",
            labels={'sub_category': 'Sub-Category', 'price $': 'Price ($)'}
        )
        fig.update_layout(xaxis_tickangle=45)
        story.append(fig)
    story.append(Spacer(1, 12))

    top_products = kd_df.nlargest(10, 'bought in past month')[['asin', 'bought in past month']]
    if not top_products.empty:
        fig = px.bar(
            top_products,
            y='asin',
            x='bought in past month',
            orientation='h',
            title=f"Top 10 Products by Bought in Past Month in {category_to_analyze}",
            labels={'asin': 'Product ASIN', 'bought in past month': 'Bought in Past Month'},
            text_auto='d'
        )
        story.append(fig)
    story.append(Spacer(1, 12))

    if 'brand' in kd_df.columns:
        brand_revenue = kd_df.groupby('brand')['revenue'].sum().sort_values(ascending=False).head(10)
        if not brand_revenue.empty:
            brand_revenue_df = brand_revenue.reset_index()
            fig = px.bar(
                brand_revenue_df,
                x='brand',
                y='revenue',
                title=f"Top 10 Brands by Revenue in {category_to_analyze}",
                labels={'brand': 'Brand', 'revenue': 'Revenue ($)'},
                text_auto='.2s'
            )
            fig.update_layout(xaxis_tickangle=45)
            story.append(fig)
    story.append(Spacer(1, 12))

    if 'date' in kd_df.columns:
        sub_price_time_data = kd_df[kd_df['sub_category'].isin(top_10_subcategories)].groupby(['date', 'sub_category'])['price $'].median().unstack().dropna(how='all')
        sub_price_time_data.index = pd.to_datetime(sub_price_time_data.index)

        if not sub_price_time_data.empty:
            for sub_cat in sub_price_time_data.columns:
                valid_data = sub_price_time_data[sub_cat].dropna()
                if not valid_data.empty:
                    fig = px.line(
                        x=valid_data.index,
                        y=valid_data,
                        title=f"Median Price Trend for Sub-Category '{sub_cat}' in {category_to_analyze}",
                        labels={'x': 'Date', 'y': 'Median Price ($)'}
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    story.append(fig)
                    story.append(Spacer(1, 12))

        top_brands = kd_df.groupby('brand')['revenue'].sum().nlargest(5).index
        for brand in top_brands:
            brand_data = kd_df[kd_df['brand'] == brand].groupby('date').agg({
                'price $': 'median', 
                'bought in past month': 'sum', 
                'revenue': 'sum'
            }).reset_index()
            brand_data['date'] = pd.to_datetime(brand_data['date'])
            brand_data = brand_data.dropna()

            significant_change = any([
                has_significant_change(brand_data['price $']),
                has_significant_change(brand_data['bought in past month']),
                has_significant_change(brand_data['revenue'])
            ])

            if significant_change and not brand_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=brand_data['date'],
                    y=brand_data['price $'],
                    mode='lines+markers',
                    name='Median Price ($)',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=brand_data['date'],
                    y=brand_data['bought in past month'],
                    mode='lines+markers',
                    name='Bought in Past Month',
                    yaxis='y2',
                    line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=brand_data['date'],
                    y=brand_data['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    yaxis='y3',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title=f"Trends Over Time for Brand '{brand}' in {category_to_analyze}",
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Median Price ($)', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                    yaxis2=dict(title='Bought in Past Month', titlefont=dict(color='orange'), tickfont=dict(color='orange'), overlaying='y', side='right'),
                    yaxis3=dict(title='Revenue', titlefont=dict(color='green'), tickfont=dict(color='green'), overlaying='y', side='right', anchor='free', position=0.95),
                    xaxis_tickangle=45
                )
                story.append(fig)
                story.append(Spacer(1, 12))

        top_asins = kd_df.nlargest(5, 'bought in past month')['asin'].tolist()
        for asin in top_asins:
            asin_data = kd_df[kd_df['asin'] == asin].groupby('date').agg({
                'price $': 'median', 
                'bought in past month': 'sum', 
                'revenue': 'sum'
            }).reset_index()
            asin_data['date'] = pd.to_datetime(asin_data['date'])
            asin_data = asin_data.dropna()

            significant_change = any([
                has_significant_change(asin_data['price $']),
                has_significant_change(asin_data['bought in past month']),
                has_significant_change(asin_data['revenue'])
            ])

            if significant_change and not asin_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=asin_data['date'],
                    y=asin_data['price $'],
                    mode='lines+markers',
                    name='Median Price ($)',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=asin_data['date'],
                    y=asin_data['bought in past month'],
                    mode='lines+markers',
                    name='Bought in Past Month',
                    yaxis='y2',
                    line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=asin_data['date'],
                    y=asin_data['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    yaxis='y3',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title=f"Trends Over Time for ASIN '{asin}' in {category_to_analyze}",
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Median Price ($)', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                    yaxis2=dict(title='Bought in Past Month', titlefont=dict(color='orange'), tickfont=dict(color='orange'), overlaying='y', side='right'),
                    yaxis3=dict(title='Revenue', titlefont=dict(color='green'), tickfont=dict(color='green'), overlaying='y', side='right', anchor='free', position=0.95),
                    xaxis_tickangle=45
                )
                story.append(fig)
                story.append(Spacer(1, 12))

    hk_insights = ai.generate_category_insights(kd_df, category_to_analyze, top_10_subcategories)
    story.append(Paragraph(f"Nhận Xét Về Category '{category_to_analyze}':", styles['VietnameseHeading3']))
    
    for insight_group in hk_insights:
        main_insight = insight_group[0]
        asin_insights = insight_group[1] if len(insight_group) > 1 else []

        story.append(Paragraph(main_insight, styles['VietnameseNormal']))

        if asin_insights:
            story.append(Paragraph("Trong đó, các sản phẩm nổi bật bao gồm:", styles['VietnameseNormal']))
            for asin_insight in asin_insights:
                story.append(Paragraph(f"• {asin_insight}", styles['VietnameseNormal']))

        mentioned_asins = set()
        for asin in kd_df['asin'].unique():
            if any(asin in str(insight) for insight in [main_insight] + asin_insights):
                mentioned_asins.add(asin)

        asin_bought_pairs = [(asin, kd_df[kd_df['asin'] == asin]['bought in past month'].iloc[0]) for asin in mentioned_asins]
        asin_bought_pairs = sorted(asin_bought_pairs, key=lambda x: x[1], reverse=True)
        mentioned_asins_sorted = [pair[0] for pair in asin_bought_pairs]

        if mentioned_asins_sorted:
            image_elements = []
            image_urls = []
            for asin in mentioned_asins_sorted:
                image_url = kd_df[kd_df['asin'] == asin]['image url'].iloc[0] if 'image url' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['image url'].iloc[0]) else None
                if image_url:
                    image_urls.append(image_url)
                else:
                    image_urls.append(None)

            images = asyncio.run(download_images(image_urls))

            for idx, asin in enumerate(mentioned_asins_sorted):
                image_buf = images[idx]
                brand = kd_df[kd_df['asin'] == asin]['brand'].iloc[0] if 'brand' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['brand'].iloc[0]) else "không xác định"
                
                asin_data = kd_df[kd_df['asin'] == asin].sort_values(by='date', ascending=False)
                price = asin_data['price $'].iloc[0] if not asin_data.empty else 0.0
                
                url = kd_df[kd_df['asin'] == asin]['url'].iloc[0] if 'url' in kd_df.columns and not pd.isna(kd_df[kd_df['asin'] == asin]['url'].iloc[0]) else f"https://www.amazon.com/dp/{asin}"

                trend = get_price_trend(asin, unique_df)
                trend_symbol = ""
                if trend == "up":
                    trend_symbol = '<span style="color:green; font-weight:bold;">↑</span>'
                elif trend == "down":
                    trend_symbol = '<span style="color:red; font-weight:bold;">↓</span>'

                caption_text = f"ASIN {asin} - Thương hiệu {brand} - Giá ${price:.2f}"
                caption = (caption_text, url, trend_symbol)

                if image_buf:
                    image_elements.append([image_buf, caption])
                else:
                    image_elements.append([Paragraph(f"Hình ảnh không khả dụng cho ASIN {asin}", styles['VietnameseNormal']), Spacer(1, 12)])

            if image_elements:
                image_table = []
                row = []
                for i, elem in enumerate(image_elements):
                    row.append(elem)
                    if (i + 1) % 3 == 0 or (i + 1) == len(image_elements):
                        image_table.append(row)
                        row = []
                if row:
                    image_table.append(row)
                story.append(Table(image_table, colWidths=[150] * 3))
                story.append(Spacer(1, 12))

    return story

def main(unique_df, top_subcategories, category_to_analyze, categories_to_analyze):
    return analyze_category(unique_df, top_subcategories, category_to_analyze, categories_to_analyze)