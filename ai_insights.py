import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
import json
import os
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

class AIInsights:
    def __init__(self, knowledge_file='knowledge.json'):
        self.knowledge_file = knowledge_file
        self.knowledge = self.load_knowledge()
        self.clusters = None
        self.regressor = None
        self.classifier = None
        self.mentioned_asins = set()

    def load_knowledge(self):
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r') as f:
                data = json.load(f)
                if 'time_series_insights' not in data:
                    data['time_series_insights'] = {}
                return data
        return {
            'insights': [],
            'patterns': {},
            'category_performance': {},
            'time_series_insights': {}
        }

    def save_knowledge(self):
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge, f, indent=4)

    def preprocess_data(self, df):
        if 'date' in df.columns:
            df = df.sort_values(by=['asin', 'date'], ascending=[True, False])
        unique_df = df.drop_duplicates(subset=['asin'], keep='first')
        stats = {
            'total_revenue': unique_df['revenue'].sum() if 'revenue' in unique_df.columns else 0,
            'avg_revenue': unique_df['revenue'].mean() if 'revenue' in unique_df.columns else 0,
            'total_sales_bought': unique_df['bought in past month'].sum() if 'bought in past month' in unique_df.columns else 0,
            'median_price': unique_df['price $'].median() if 'price $' in unique_df.columns else 0
        }

        cat_stats = unique_df.groupby('category')['bought in past month'].sum().sort_values(ascending=False)
        sub_stats = unique_df.groupby('sub_category')['bought in past month'].sum().sort_values(ascending=False)
        sub_stats_top10 = sub_stats.head(10)

        cat_stats_total = cat_stats.sum()
        cat_stats_percent = (cat_stats / cat_stats_total * 100) if cat_stats_total > 0 else pd.Series()

        sub_stats_total = sub_stats.sum()
        sub_stats_percent = (sub_stats / sub_stats_total * 100) if sub_stats_total > 0 else pd.Series()

        return {
            'stats': stats,
            'cat_stats': cat_stats,
            'sub_stats': sub_stats,
            'sub_stats_top10': sub_stats_top10,
            'cat_stats_percent': cat_stats_percent,
            'sub_stats_percent': sub_stats_percent,
            'unique_df': unique_df
        }

    def detect_anomalies(self, df, metric='revenue'):
        if metric not in df.columns:
            return None
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        data = df[[metric]].fillna(0)
        anomalies = iso_forest.fit_predict(data)
        return df[anomalies == -1]

    def forecast_trend(self, df, metric='bought in past month'):
        if 'date' not in df.columns or metric not in df.columns:
            return None
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        prophet_df = df.groupby('date')[metric].sum().reset_index().rename(columns={'date': 'ds', metric: 'y'})
        if len(prophet_df) < 2:
            return None
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    def cluster_analysis(self, df):
        if 'category' not in df.columns or 'bought in past month' not in df.columns:
            return None

        cluster_data = df.groupby('category').agg({
            'bought in past month': 'sum',
            'revenue': 'sum',
            'price $': 'mean'
        }).fillna(0)

        cluster_data_normalized = (cluster_data - cluster_data.mean()) / cluster_data.std()
        cluster_data_normalized = cluster_data_normalized.fillna(0)

        n_samples = cluster_data_normalized.shape[0]
        max_clusters = 3
        n_clusters = max(1, min(max_clusters, n_samples))

        if n_samples < 2:
            clusters = np.zeros(n_samples, dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(cluster_data_normalized)

        self.clusters = pd.DataFrame({
            'category': cluster_data.index,
            'cluster': clusters,
            'bought_in_past_month': cluster_data['bought in past month'],
            'revenue': cluster_data['revenue'],
            'avg_price': cluster_data['price $']
        })

        return self.clusters

    def trend_analysis(self, df):
        if 'date' not in df.columns or 'bought in past month' not in df.columns:
            return None

        df = df.copy()
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        trend_data = df.groupby('date')['bought in past month'].sum().reset_index()

        X = np.array(range(len(trend_data))).reshape(-1, 1)
        y = trend_data['bought in past month'].values

        self.regressor = LinearRegression()
        self.regressor.fit(X, y)

        trend_slope = self.regressor.coef_[0]
        return trend_slope

    def time_series_analysis(self, df, metric='bought in past month', group_by='category'):
        if 'date' not in df.columns or metric not in df.columns:
            return None, None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        time_data = df.groupby(['date', group_by])[metric].sum().unstack().fillna(0)

        insights = []
        forecasts = {}
        for entity in time_data.columns:
            series = time_data[entity]
            if len(series) > 1:
                X = np.array(range(len(series))).reshape(-1, 1)
                y = series.values
                model = LinearRegression()
                model.fit(X, y)
                trend = model.coef_[0]
                forecast = model.predict(np.array([[len(series)], [len(series) + 1], [len(series) + 2]]))
                insights.append(
                    f"• {group_by.capitalize()} '{entity}': {metric} có xu hướng {'tăng' if trend > 0 else 'giảm'} với tốc độ {abs(trend):,.2f} mỗi ngày. Dự đoán 3 ngày tới: {forecast.mean():,.2f}."
                )
                forecasts[entity] = forecast.tolist()
        
        self.knowledge['time_series_insights'][metric] = insights
        return insights, forecasts

    def classify_potential(self, df):
        if 'category' not in df.columns or 'bought in past month' not in df.columns:
            return None

        features = df.groupby('category').agg({
            'bought in past month': 'sum',
            'revenue': 'sum',
            'price $': 'mean'
        }).fillna(0)

        labels = (features['bought in past month'] > features['bought in past month'].median()) & \
                 (features['revenue'] > features['revenue'].median())
        labels = labels.astype(int)

        self.classifier = DecisionTreeClassifier(random_state=42)
        self.classifier.fit(features, labels)

        predictions = self.classifier.predict(features)
        return pd.DataFrame({'category': features.index, 'potential': predictions})

    def generate_insights(self, df):
        insights = []
        data = self.preprocess_data(df)
        stats = data['stats']
        cat_stats_percent = data['cat_stats_percent']
        sub_stats_percent = data['sub_stats_percent']
        unique_df = data['unique_df']

        forecast = self.forecast_trend(unique_df, 'bought in past month')
        if forecast is not None:
            latest_forecast = forecast.iloc[-1]
            insights.append(
                f"• Dự báo: Số lượng mua trong 7 ngày tới được dự đoán khoảng {latest_forecast['yhat']:,.0f} sản phẩm."
            )

        if not cat_stats_percent.empty:
            top_cat = cat_stats_percent.index[0]
            top_cat_percent = cat_stats_percent.iloc[0]
            top_cat_revenue = unique_df[unique_df['category'] == top_cat]['revenue'].sum()
            top_cat_avg_price = unique_df[unique_df['category'] == top_cat]['price $'].mean()
            market_avg_price = stats['median_price']

            if top_cat_percent > (cat_stats_percent.mean() + cat_stats_percent.std()):
                insights.append(
                    f"• Category '{top_cat}' đang dẫn đầu với {top_cat_percent:.1f}% tổng số lượng mua, mang về ${top_cat_revenue:,.2f} doanh thu. Giá trung bình ${top_cat_avg_price:.2f}, {'cao hơn' if top_cat_avg_price > market_avg_price else 'thấp hơn'} giá thị trường ${market_avg_price:.2f}."
                )

        if not sub_stats_percent.empty:
            top_sub = sub_stats_percent.index[0]
            top_sub_percent = sub_stats_percent.iloc[0]
            sub_data = unique_df[unique_df['sub_category'] == top_sub][['price $', 'bought in past month']]
            correlation = sub_data['price $'].corr(sub_data['bought in past month']) if len(sub_data) > 1 else 0

            insights.append(
                f"• Sub-category '{top_sub}' bán chạy nhất với {top_sub_percent:.1f}% tổng số lượng mua. Mối liên hệ giá và số lượng mua: {'giá cao thì bán tốt' if correlation > 0.3 else 'giá thấp thì bán tốt' if correlation < -0.3 else 'giá không ảnh hưởng nhiều'}."
            )

        clusters = self.cluster_analysis(unique_df)
        if clusters is not None:
            for cluster_id in clusters['cluster'].unique():
                cluster_data = clusters[clusters['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    avg_sales = cluster_data['bought_in_past_month'].mean()
                    avg_revenue = cluster_data['revenue'].mean()
                    avg_price = cluster_data['avg_price'].mean()
                    cluster_categories = cluster_data['category'].tolist()
                    insights.append(
                        f"• Nhóm category {', '.join(cluster_categories)}: Số lượng mua trung bình {avg_sales:,.0f}, doanh thu trung bình ${avg_revenue:,.2f}, giá trung bình ${avg_price:.2f}."
                    )

        trend_slope = self.trend_analysis(unique_df)
        if trend_slope is not None:
            insights.append(
                f"• Xu hướng mua hàng {'tăng' if trend_slope > 0 else 'giảm'} với tốc độ {abs(trend_slope):,.0f} sản phẩm mỗi tháng."
            )

        potential = self.classify_potential(unique_df)
        if potential is not None:
            potential_categories = potential[potential['potential'] == 1]['category'].tolist()
            if potential_categories:
                insights.append(
                    f"• Các category tiềm năng cao: {', '.join(potential_categories)}."
                )

        for metric in ['bought in past month', 'price $', 'revenue']:
            ts_insights, _ = self.time_series_analysis(unique_df, metric, group_by='category')
            if ts_insights:
                insights.extend(ts_insights)

        if 'date' in unique_df.columns:
            top_asins = unique_df.nlargest(10, 'bought in past month')['asin'].tolist()
            for asin in top_asins:
                changes = self.calculate_changes(unique_df, 'asin', asin)
                price_change = changes['price $']
                bought_change = changes['bought in past month']
                revenue_change = changes['revenue']

                change_comments = []
                analysis_comments = []

                if price_change['latest_date']:
                    price_diff = price_change['change']
                    price_percent = price_change['percent_change']
                    if abs(price_diff) >= 0.01:
                        change_comments.append(
                            f"giá ${price_change['latest_value']:.2f} đã {'tăng' if price_diff > 0 else 'giảm'} ${abs(price_diff):.2f} so với ngày {price_change['previous_date']} ({abs(price_percent):.2f}%)"
                        )
                        if price_diff > 0:
                            analysis_comments.append("Tăng giá có thể ảnh hưởng đến số lượng bán.")
                        else:
                            analysis_comments.append("Giảm giá có thể thu hút thêm khách hàng.")

                if bought_change['latest_date']:
                    bought_diff = bought_change['change']
                    bought_percent = bought_change['percent_change']
                    if abs(bought_diff) > 0:
                        change_comments.append(
                            f"số lượng mua {bought_change['latest_value']:,.0f} sản phẩm đã {'tăng' if bought_diff > 0 else 'giảm'} {abs(bought_percent):.2f}% so với ngày {bought_change['previous_date']}"
                        )
                        if bought_diff > 0:
                            analysis_comments.append("Sản phẩm đang được ưa chuộng hơn.")
                        else:
                            analysis_comments.append("Cần xem xét yếu tố giá hoặc cạnh tranh.")

                if revenue_change['latest_date']:
                    revenue_diff = revenue_change['change']
                    revenue_percent = revenue_change['percent_change']
                    if abs(revenue_diff) > 0:
                        change_comments.append(
                            f"doanh thu ${revenue_change['latest_value']:,.2f} đã {'tăng' if revenue_diff > 0 else 'giảm'} {abs(revenue_percent):.2f}% so với ngày {revenue_change['previous_date']}"
                        )
                        if revenue_diff < 0 and price_diff > 0:
                            analysis_comments.append("Doanh thu giảm sau tăng giá, xem xét điều chỉnh giá.")
                        elif revenue_diff > 0 and price_diff < 0:
                            analysis_comments.append("Giảm giá đang mang lại hiệu quả doanh thu.")

                if change_comments:
                    product_data = unique_df[unique_df['asin'] == asin].iloc[0]
                    brand = product_data['brand'] if 'brand' in unique_df.columns and not pd.isna(product_data['brand']) else "không xác định"
                    creation_date = product_data['creation date'] if 'creation date' in unique_df.columns and not pd.isna(product_data['creation date']) else "Không có dữ liệu"
                    variation = product_data['variation'] if 'variation' in unique_df.columns and not pd.isna(product_data['variation']) else "Không có dữ liệu"
                    ratings = product_data['ratings'] if 'ratings' in unique_df.columns and not pd.isna(product_data['ratings']) else "Không có dữ liệu"
                    review_count = product_data['review count'] if 'review count' in unique_df.columns and not pd.isna(product_data['review count']) else "Không có dữ liệu"
                    seller_country = product_data['seller country/region'] if 'seller country/region' in unique_df.columns and not pd.isna(product_data['seller country/region']) else "Không có dữ liệu"

                    main_comment = (
                        f"• ASIN '{asin}' thuộc thương hiệu {brand} có giá ${product_data['price $']:.2f}, "
                        f"bán được {product_data['bought in past month']:,.0f} sản phẩm, doanh thu ${product_data['revenue']:,.2f}. "
                        f"Ngày ra mắt: {creation_date}, Số lượng biến thể: {variation}, "
                        f"Điểm đánh giá: {ratings}, Số lượng đánh giá: {review_count}, Thuộc seller từ: {seller_country}"
                    )
                    if change_comments:
                        main_comment += f". Thay đổi: {', '.join(change_comments)}"
                    if analysis_comments:
                        main_comment += f". Phân tích: {', '.join(analysis_comments)}"

                    insights.append(main_comment)

        return insights

    def calculate_changes(self, df, identifier, identifier_value, metrics=['price $', 'bought in past month', 'revenue']):
        if 'date' not in df.columns:
            return {metric: {'change': 0, 'percent_change': 0, 'latest_date': None, 'previous_date': None, 'latest_value': None, 'previous_value': None} for metric in metrics}

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        data = df[df[identifier] == identifier_value].groupby('date').agg({metric: 'mean' for metric in metrics}).reset_index()
        data = data.sort_values('date')

        changes = {}
        if len(data) >= 2:
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            latest_date = latest['date'].strftime('%Y-%m-%d')
            previous_date = previous['date'].strftime('%Y-%m-%d')

            for metric in metrics:
                latest_value = latest[metric]
                previous_value = previous[metric]
                change = latest_value - previous_value
                percent_change = ((latest_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
                changes[metric] = {
                    'change': change,
                    'percent_change': percent_change,
                    'latest_date': latest_date,
                    'previous_date': previous_date,
                    'latest_value': latest_value,
                    'previous_value': previous_value
                }
        else:
            changes = {metric: {'change': 0, 'percent_change': 0, 'latest_date': None, 'previous_date': None, 'latest_value': None, 'previous_value': None} for metric in metrics}

        return changes

    def generate_category_insights(self, df, category, top_10_subcategories=None):
        insights = []
        self.mentioned_asins.clear()
        if 'date' in df.columns:
            df = df.sort_values(by=['asin', 'date'], ascending=[True, False])
        unique_df = df.drop_duplicates(subset=['asin'], keep='first')
        cat_df = unique_df[unique_df['category'] == category]

        if cat_df.empty:
            return [[f"• Không có dữ liệu cho category '{category}'."]]

        cat_revenue = cat_df['revenue'].sum()
        cat_sales_bought = cat_df['bought in past month'].sum()
        cat_median_price = cat_df['price $'].median()
        insights.append([f"• Category '{category}' có tổng doanh thu là ${cat_revenue:,.2f} và tổng số lượng mua trong tháng qua là {cat_sales_bought:,.0f} sản phẩm."])

        sub_stats = cat_df.groupby('sub_category').agg({
            'revenue': 'sum',
            'price $': ['median', 'min', 'max'],
            'bought in past month': 'sum'
        }).fillna(0)

        if not sub_stats.empty:
            sub_stats.columns = ['revenue', 'median_price', 'min_price', 'max_price', 'bought_in_past_month']
            
            if top_10_subcategories is not None:
                valid_subcategories = [sub for sub in top_10_subcategories if sub in sub_stats.index]
                if not valid_subcategories:
                    top_sub_stats = sub_stats.nlargest(10, 'revenue')
                else:
                    top_sub_stats = sub_stats.loc[valid_subcategories]
            else:
                top_sub_stats = sub_stats.nlargest(10, 'revenue')

            if 'date' in cat_df.columns:
                sub_price_time_data = cat_df[cat_df['sub_category'].isin(top_sub_stats.index)].groupby(['date', 'sub_category'])['price $'].median().unstack().dropna(how='all')
                sub_price_time_data.index = pd.to_datetime(sub_price_time_data.index)

            if not top_sub_stats.empty:
                for sub_cat in top_sub_stats.index:
                    sub_revenue = top_sub_stats.loc[sub_cat, 'revenue']
                    sub_sales = top_sub_stats.loc[sub_cat, 'bought_in_past_month']
                    median_price = top_sub_stats.loc[sub_cat, 'median_price']
                    min_price = top_sub_stats.loc[sub_cat, 'min_price']
                    max_price = top_sub_stats.loc[sub_cat, 'max_price']

                    price_trend_comment = ""
                    if 'date' in cat_df.columns and sub_cat in sub_price_time_data.columns:
                        price_series = sub_price_time_data[sub_cat].dropna()
                        if len(price_series) > 1:
                            price_change = price_series.iloc[-1] - price_series.iloc[0]
                            price_percent_change = (price_change / price_series.iloc[0] * 100) if price_series.iloc[0] != 0 else 0
                            trend_direction = "tăng" if price_change > 0 else "giảm"
                            price_trend_comment = (
                                f"Giá trung vị {trend_direction} ${abs(price_change):,.2f} ({abs(price_percent_change):,.2f}%) "
                                f"từ ${price_series.iloc[0]:,.2f} xuống ${price_series.iloc[-1]:,.2f}."
                            )

                    sorted_cat_df = cat_df.sort_values('bought in past month', ascending=False)
                    top_products = sorted_cat_df[cat_df['sub_category'] == sub_cat].nlargest(10, 'bought in past month')[['asin', 'bought in past month', 'revenue', 'price $', 'creation date', 'variation', 'ratings', 'review count', 'seller country/region', 'brand']]
                    if not top_products.empty:
                        asin_comments = []
                        raw_cat_df = df[df['category'] == category]
                        for idx, product in top_products.iterrows():
                            asin = product['asin']
                            self.mentioned_asins.add(asin)
                            bought = product['bought in past month']
                            revenue = product['revenue']
                            if 'date' in raw_cat_df.columns:
                                asin_data = raw_cat_df[raw_cat_df['asin'] == asin].sort_values('date', ascending=False)
                                price = asin_data['price $'].iloc[0] if not asin_data.empty else product['price $']
                            else:
                                price = product['price $']
                            brand = product['brand'] if 'brand' in raw_cat_df.columns and not pd.isna(product['brand']) else "không xác định"
                            creation_date = product['creation date'] if 'creation date' in raw_cat_df.columns and not pd.isna(product['creation date']) else "Không có dữ liệu"
                            variation = product['variation'] if 'variation' in raw_cat_df.columns and not pd.isna(product['variation']) else "Không có dữ liệu"
                            ratings = product['ratings'] if 'ratings' in raw_cat_df.columns and not pd.isna(product['ratings']) else "Không có dữ liệu"
                            review_count = product['review count'] if 'review count' in raw_cat_df.columns and not pd.isna(product['review count']) else "Không có dữ liệu"
                            seller_country = product['seller country/region'] if 'seller country/region' in raw_cat_df.columns and not pd.isna(product['seller country/region']) else "Không có dữ liệu"

                            changes = self.calculate_changes(raw_cat_df, 'asin', asin)
                            price_change = changes['price $']
                            bought_change = changes['bought in past month']
                            revenue_change = changes['revenue']

                            change_comments = []
                            analysis_comments = []

                            if price_change['latest_date']:
                                price_diff = price_change['change']
                                price_percent = price_change['percent_change']
                                if abs(price_diff) >= 0.01:
                                    change_comments.append(
                                        f"giá ${price_change['latest_value']:.2f} đã {'tăng' if price_diff > 0 else 'giảm'} ${abs(price_diff):.2f} so với ngày {price_change['previous_date']} ({abs(price_percent):.2f}%)"
                                    )
                                    if price_diff > 0:
                                        analysis_comments.append("Tăng giá có thể ảnh hưởng đến số lượng bán")
                                    else:
                                        analysis_comments.append("Giảm giá có thể thu hút thêm khách hàng")

                            if bought_change['latest_date']:
                                bought_diff = bought_change['change']
                                bought_percent = bought_change['percent_change']
                                if abs(bought_diff) > 0:
                                    change_comments.append(
                                        f"số lượng mua {bought_change['latest_value']:,.0f} sản phẩm đã {'tăng' if bought_diff > 0 else 'giảm'} {abs(bought_percent):.2f}% so với ngày {bought_change['previous_date']}"
                                    )
                                    if bought_diff > 0:
                                        analysis_comments.append("Sản phẩm đang được ưa chuộng hơn")
                                    else:
                                        analysis_comments.append("Cần xem xét yếu tố giá hoặc cạnh tranh")

                            if revenue_change['latest_date']:
                                revenue_diff = revenue_change['change']
                                revenue_percent = revenue_change['percent_change']
                                if abs(revenue_diff) > 0:
                                    change_comments.append(
                                        f"doanh thu ${revenue_change['latest_value']:,.2f} đã {'tăng' if revenue_diff > 0 else 'giảm'} {abs(revenue_percent):.2f}% so với ngày {revenue_change['previous_date']}"
                                    )
                                    if revenue_diff < 0 and price_diff > 0:
                                        analysis_comments.append("Doanh thu giảm sau tăng giá, xem xét điều chỉnh giá")
                                    elif revenue_diff > 0 and price_diff < 0:
                                        analysis_comments.append("Giảm giá đang mang lại hiệu quả doanh thu")

                            price_change_text = ""
                            if price_change['latest_date'] and abs(price_diff) >= 0.01:
                                price_change_text = f" ({'tăng' if price_diff > 0 else 'giảm'} ${abs(price_diff):.2f})"

                            main_comment = (
                                f"ASIN '{asin}' thuộc thương hiệu {brand} có giá ${price:.2f}{price_change_text}, "
                                f"bán được {bought:,.0f} sản phẩm, doanh thu ${revenue:,.2f}. "
                                f"Ngày ra mắt: {creation_date}, Số lượng biến thể: {variation}, "
                                f"Điểm đánh giá: {ratings}, Số lượng đánh giá: {review_count}, "
                                f"Thuộc seller từ: {seller_country}"
                            )
                            if change_comments:
                                main_comment += f". Thay đổi: {', '.join(change_comments)}"
                            if analysis_comments:
                                main_comment += f". Phân tích: {', '.join(analysis_comments)}"

                            asin_comments.append(main_comment)

                        main_insight = (
                            f"• Sub-category '{sub_cat}' có doanh thu khoảng ${sub_revenue/1_000_000:.1f}M và số lượng mua {sub_sales:,.0f} sản phẩm. "
                            f"Giá từ ${min_price:.2f} đến ${max_price:.2f}, trung vị ${median_price:.2f}."
                        )
                        if price_trend_comment:
                            main_insight += f" {price_trend_comment}"
                        insights.append([main_insight, asin_comments])
                    else:
                        main_insight = (
                            f"• Sub-category '{sub_cat}' có doanh thu khoảng ${sub_revenue/1_000_000:.1f}M và số lượng mua {sub_sales:,.0f} sản phẩm. "
                            f"Giá từ ${min_price:.2f} đến ${max_price:.2f}, trung vị ${median_price:.2f}. "
                            f"Không có sản phẩm nổi bật."
                        )
                        if price_trend_comment:
                            main_insight += f" {price_trend_comment}"
                        insights.append([main_insight])

        return insights

    def analyze_keyword_ranking(self, raw_df, asin):
        """
        Phân tích organic rank và sponsored rank theo keyword cho một ASIN.
        Trả về danh sách insights, bảng ranking, hai biểu đồ (organic và sponsored), và nhận xét.
        """
        if raw_df is None or 'keyword' not in raw_df.columns or 'date' not in raw_df.columns:
            return [], None, None, None, []

        # Lọc dữ liệu theo ASIN
        asin_data = raw_df[raw_df['asin'] == asin].copy()
        if asin_data.empty:
            return [], None, None, None, []

        # Đảm bảo cột date ở định dạng datetime
        asin_data['date'] = pd.to_datetime(asin_data['date'])
        asin_data = asin_data.sort_values('date')

        # Thay NaN bằng 0 cho organic rank và sponsored rank
        asin_data['organic rank'] = asin_data['organic rank'].fillna(0).astype(int)
        asin_data['sponsored rank'] = asin_data['sponsored rank'].fillna(0).astype(int)

        # Lấy danh sách keyword duy nhất
        keywords = asin_data['keyword'].unique()
        if not keywords.size:
            return [], None, None, None, []

        insights = []
        ranking_data = []
        comments = []

        # Tạo hai biểu đồ
        organic_fig = go.Figure()
        sponsored_fig = go.Figure()

        for keyword in keywords:
            keyword_data = asin_data[asin_data['keyword'] == keyword][['date', 'organic rank', 'sponsored rank']]
            if len(keyword_data) < 1:
                continue

            # Tính thay đổi rank
            latest_data = keyword_data.iloc[-1]
            latest_date = latest_data['date'].strftime('%Y-%m-%d')
            latest_organic = latest_data['organic rank']
            latest_sponsored = latest_data['sponsored rank']

            organic_change = None
            sponsored_change = None
            organic_comment = ""
            sponsored_comment = ""

            if len(keyword_data) >= 2:
                previous_data = keyword_data.iloc[-2]
                previous_date = previous_data['date'].strftime('%Y-%m-%d')
                previous_organic = previous_data['organic rank']
                previous_sponsored = previous_data['sponsored rank']

                # Organic rank: thấp hơn là tốt hơn (ví dụ: từ 10 xuống 5 là cải thiện)
                if latest_organic != previous_organic and latest_organic != 0:
                    organic_change = previous_organic - latest_organic
                    organic_comment = (
                        f"Organic rank {'cải thiện' if organic_change > 0 else 'giảm'} "
                        f"từ {previous_organic} xuống {latest_organic} "
                        f"từ ngày {previous_date} đến {latest_date}"
                    )
                elif latest_organic == 0:
                    organic_comment = "Không có organic rank"
                else:
                    organic_comment = "Organic rank không thay đổi"

                # Sponsored rank
                if latest_sponsored != previous_sponsored and latest_sponsored != 0:
                    sponsored_change = previous_sponsored - latest_sponsored
                    sponsored_comment = (
                        f"Sponsored rank {'cải thiện' if sponsored_change > 0 else 'giảm'} "
                        f"từ {previous_sponsored} xuống {latest_sponsored} "
                        f"từ ngày {previous_date} đến {latest_date}"
                    )
                elif latest_sponsored == 0:
                    sponsored_comment = "Không có sponsored rank"
                else:
                    sponsored_comment = "Sponsored rank không thay đổi"

            else:
                organic_comment = f"Organic rank hiện tại: {latest_organic}" if latest_organic != 0 else "Không có organic rank"
                sponsored_comment = f"Sponsored rank hiện tại: {latest_sponsored}" if latest_sponsored != 0 else "Không có sponsored rank"

            # Thêm insights
            insights.append(f"• Keyword '{keyword}': {organic_comment}. {sponsored_comment}.")

            # Thêm vào bảng ranking
            ranking_data.append({
                'Keyword': keyword,
                'Latest Organic Rank': latest_organic,
                'Latest Sponsored Rank': latest_sponsored,
                'Organic Change': organic_change if organic_change is not None else 0,
                'Sponsored Change': sponsored_change if sponsored_change is not None else 0,
                'Latest Date': latest_date
            })

            # Thêm vào biểu đồ organic
            organic_fig.add_trace(go.Scatter(
                x=keyword_data['date'],
                y=keyword_data['organic rank'],
                mode='lines+markers',
                name=keyword,
                line=dict(dash='solid')
            ))

            # Thêm vào biểu đồ sponsored
            sponsored_fig.add_trace(go.Scatter(
                x=keyword_data['date'],
                y=keyword_data['sponsored rank'],
                mode='lines+markers',
                name=keyword,
                line=dict(dash='solid')
            ))

            # Nhận xét chung
            if len(keyword_data) >= 2:
                organic_trend = keyword_data['organic rank'].diff().mean()
                sponsored_trend = keyword_data['sponsored rank'].diff().mean()
                if organic_trend < 0:
                    comments.append(f"• Keyword '{keyword}': Organic rank có xu hướng cải thiện.")
                elif organic_trend > 0:
                    comments.append(f"• Keyword '{keyword}': Organic rank có xu hướng giảm.")
                if sponsored_trend < 0 and latest_sponsored != 0:
                    comments.append(f"• Keyword '{keyword}': Sponsored rank có xu hướng cải thiện.")
                elif sponsored_trend > 0 and latest_sponsored != 0:
                    comments.append(f"• Keyword '{keyword}': Sponsored rank có xu hướng giảm.")

        # Cấu hình biểu đồ organic
        organic_fig.update_layout(
            title=f"Xu hướng Organic Rank của ASIN '{asin}' theo Keyword",
            xaxis=dict(title='Ngày'),
            yaxis=dict(title='Organic Rank (thấp hơn là tốt hơn)', autorange='reversed'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_tickangle=45
        )

        # Cấu hình biểu đồ sponsored
        sponsored_fig.update_layout(
            title=f"Xu hướng Sponsored Rank của ASIN '{asin}' theo Keyword",
            xaxis=dict(title='Ngày'),
            yaxis=dict(title='Sponsored Rank (thấp hơn là tốt hơn)', autorange='reversed'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_tickangle=45
        )

        ranking_df = pd.DataFrame(ranking_data) if ranking_data else None
        return insights, ranking_df, organic_fig, sponsored_fig, comments

    def analyze_asin(self, unique_df, raw_df, asin):
        """
        Phân tích chi tiết một ASIN, bao gồm brand, creation date, seller, price, bought, và keyword ranking.
        Trả về danh sách các phần tử (văn bản, bảng, biểu đồ) để hiển thị.
        """
        result = []

        # Lấy thông tin từ unique_df (bản ghi mới nhất)
        asin_data = unique_df[unique_df['asin'] == asin]
        if asin_data.empty:
            return [f"• ASIN '{asin}' không có trong dữ liệu."]

        product = asin_data.iloc[0]
        brand = product['brand'] if 'brand' in unique_df.columns and not pd.isna(product['brand']) else "không xác định"
        creation_date = product['creation date'] if 'creation date' in unique_df.columns and not pd.isna(product['creation date']) else "Không có dữ liệu"
        seller_country = product['seller country/region'] if 'seller country/region' in unique_df.columns and not pd.isna(product['seller country/region']) else "Không có dữ liệu"
        price = product['price $'] if 'price $' in unique_df.columns else 0.0
        bought = product['bought in past month'] if 'bought in past month' in unique_df.columns else 0
        revenue = product['revenue'] if 'revenue' in unique_df.columns else 0.0

        # Thông tin cơ bản
        result.append(
            f"• **Thương hiệu**: {brand}\n"
            f"• **Ngày ra mắt**: {creation_date}\n"
            f"• **Seller từ**: {seller_country}\n"
            f"• **Giá hiện tại**: ${price:.2f}\n"
            f"• **Số lượng mua trong tháng qua**: {bought:,.0f} sản phẩm\n"
            f"• **Doanh thu**: ${revenue:,.2f}"
        )

        # Phân tích thay đổi giá
        result.append("### Phân tích Giá")
        changes = self.calculate_changes(raw_df, 'asin', asin)
        price_change = changes['price $']
        if price_change['latest_date'] and abs(price_change['change']) >= 0.01:
            price_diff = price_change['change']
            price_percent = price_change['percent_change']
            result.append(
                f"• **Thay đổi giá**: Giá ${price_change['latest_value']:.2f} đã {'tăng' if price_diff > 0 else 'giảm'} "
                f"${abs(price_diff):.2f} ({abs(price_percent):.2f}%) so với ngày {price_change['previous_date']}."
            )
            if price_diff > 0:
                result.append("  - **Phân tích**: Tăng giá có thể ảnh hưởng đến số lượng bán.")
            else:
                result.append("  - **Phân tích**: Giảm giá có thể thu hút thêm khách hàng.")

        # Biểu đồ thay đổi giá
        if 'date' in raw_df.columns and 'price $' in raw_df.columns:
            price_data = raw_df[raw_df['asin'] == asin][['date', 'price $']].copy()
            price_data['date'] = pd.to_datetime(price_data['date'])
            price_data = price_data.sort_values('date')
            if len(price_data) > 0:
                price_fig = px.line(
                    price_data,
                    x='date',
                    y='price $',
                    title=f"Xu hướng Giá của ASIN '{asin}'",
                    labels={'date': 'Ngày', 'price $': 'Giá ($)'},
                )
                price_fig.update_layout(
                    xaxis_tickangle=45,
                    xaxis=dict(tickformat='%Y-%m-%d'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                result.append("#### Biểu đồ xu hướng Giá")
                result.append(price_fig)

                # Nhận xét xu hướng giá
                if len(price_data) >= 2:
                    price_trend = price_data['price $'].diff().mean()
                    if price_trend > 0:
                        result.append("• **Nhận xét**: Giá có xu hướng tăng.")
                    elif price_trend < 0:
                        result.append("• **Nhận xét**: Giá có xu hướng giảm.")
                    else:
                        result.append("• **Nhận xét**: Giá ổn định.")

        # Phân tích keyword ranking
        ranking_insights, ranking_df, organic_fig, sponsored_fig, ranking_comments = self.analyze_keyword_ranking(raw_df, asin)
        if ranking_insights:
            result.append("### Phân tích Keyword Ranking")
            for insight in ranking_insights:
                result.append(insight)
            if ranking_df is not None:
                result.append("#### Bảng Ranking")
                result.append(ranking_df)
            if organic_fig is not None:
                result.append("#### Biểu đồ xu hướng Organic Rank")
                result.append(organic_fig)
            if sponsored_fig is not None:
                result.append("#### Biểu đồ xu hướng Sponsored Rank")
                result.append(sponsored_fig)
            if ranking_comments:
                result.append("#### Nhận xét về xu hướng")
                for comment in ranking_comments:
                    result.append(comment)

        return result