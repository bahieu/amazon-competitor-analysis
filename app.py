import streamlit as st
import pandas as pd
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from overview_report import main as overview_report_main
from category_report import main as category_report_main
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from ai_insights import AIInsights
import base64
from datetime import datetime
import asyncio
import glob
import zipfile
import shutil
from utils import download_images, get_price_trend

# Hàm tải file PDF dưới dạng link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Hàm làm sạch keyword
def clean_keyword(keyword):
    if pd.isna(keyword):
        return ""
    keyword = str(keyword).replace('\ufeff', '').strip()  # Loại bỏ BOM
    return keyword

# Hàm đọc và xử lý dữ liệu
@st.cache_data
def load_data(file_paths):
    all_dfs = []
    all_raw_dfs = []
    for file_path in file_paths:
        st.sidebar.write(f"Đang xử lý file: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'keyword' in df.columns:
                df['keyword'] = df['keyword'].apply(clean_keyword)

            required_columns = ['asin', 'category', 'sub_category', 'revenue', 'price $', 'bought in past month']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Thiếu các cột cần thiết: {', '.join(missing_columns)}")

            df['category'] = df['category'].fillna('Unknown').astype(str)
            df = df[df['asin'].str.startswith('B0', na=False)]
            
            # Lưu raw_df (chưa deduplicate) cho phân tích keyword
            raw_df = df.copy()
            all_raw_dfs.append(raw_df)

            # Deduplicate để tạo unique_df
            unique_df = df.sort_values(by=['asin', 'date'], ascending=[True, False]).drop_duplicates(subset=['asin'], keep='first')

            if 'product details' in unique_df.columns:
                unique_df['product details'] = unique_df['product details'].fillna('').astype(str)
                filtered_df = unique_df[~unique_df['product details'].str.contains('set|basket', case=False, na=False)]
            else:
                filtered_df = unique_df

            excluded_brands = ["STANLEY"]
            if 'brand' in filtered_df.columns:
                filtered_df['brand'] = filtered_df['brand'].fillna('').astype(str)
                excluded_brands_lower = [brand.lower() for brand in excluded_brands]
                final_df = filtered_df[~filtered_df['brand'].str.lower().isin(excluded_brands_lower)]
            else:
                final_df = filtered_df

            all_dfs.append(final_df)
        except Exception as e:
            st.error(f"Lỗi khi đọc dữ liệu từ {file_path}: {str(e)}")
    
    if not all_dfs:
        st.error("Không tìm thấy dữ liệu hợp lệ.")
        return None, None
    combined_unique_df = pd.concat(all_dfs, ignore_index=True)
    combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True) if all_raw_dfs else None
    return combined_unique_df, combined_raw_df

def main():
    st.set_page_config(page_title="Phân Tích Đối Thủ Amazon", layout="wide")
    st.title("Phân Tích Đối Thủ Amazon")

    # Sidebar để chọn nguồn dữ liệu
    st.sidebar.header("Nguồn Dữ Liệu")
    uploaded_files = st.sidebar.file_uploader(
        "Tải lên file CSV hoặc file ZIP chứa các file CSV",
        type=["csv", "zip"],
        accept_multiple_files=True
    )

    temp_input_dir = "temp_input"
    if not os.path.exists(temp_input_dir):
        os.makedirs(temp_input_dir)

    file_paths = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Kiểm tra kích thước file (giới hạn 100MB)
            if uploaded_file.size > 100 * 1024 * 1024:
                st.error(f"File {uploaded_file.name} quá lớn. Vui lòng tải file nhỏ hơn 100MB.")
                return
            
            file_path = os.path.join(temp_input_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Nếu là file ZIP, giải nén
            if uploaded_file.name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_input_dir)
                    extracted_files = glob.glob(os.path.join(temp_input_dir, "*.csv"))
                    file_paths.extend(extracted_files)
                    os.remove(file_path)  # Xóa file ZIP sau khi giải nén
                except zipfile.BadZipFile:
                    st.error(f"File {uploaded_file.name} không phải là file ZIP hợp lệ.")
                    return
            else:
                file_paths.append(file_path)

    if not file_paths:
        st.error("Vui lòng tải lên ít nhất một file CSV hoặc file ZIP chứa các file CSV.")
        return

    try:
        unique_df, raw_df = load_data(file_paths)
        if unique_df is None:
            return

        # Bộ lọc nâng cao
        st.sidebar.subheader("Bộ Lọc Dữ Liệu")
        if 'date' in unique_df.columns:
            unique_df = unique_df.dropna(subset=['date'])
            min_date = unique_df['date'].min().date()
            max_date = unique_df['date'].max().date()
            start_date, end_date = st.sidebar.date_input("Chọn khoảng thời gian", [min_date, max_date], min_value=min_date, max_value=max_date)
            if start_date <= end_date:
                filtered_df = unique_df[(unique_df['date'].dt.date >= start_date) & (unique_df['date'].dt.date <= end_date)]
                filtered_raw_df = raw_df[(raw_df['date'].dt.date >= start_date) & (raw_df['date'].dt.date <= end_date)] if raw_df is not None else None
            else:
                st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
                return
        else:
            filtered_df = unique_df
            filtered_raw_df = raw_df
            st.warning("Dữ liệu không có cột 'date', sử dụng toàn bộ dữ liệu.")

        if filtered_df.empty:
            st.error("Không có dữ liệu sau khi lọc theo thời gian. Vui lòng kiểm tra dữ liệu đầu vào hoặc điều chỉnh khoảng thời gian.")
            return

        if 'sub_category' in filtered_df.columns:
            filtered_df['sub_category'] = filtered_df['sub_category'].fillna('Unknown').astype(str)
            all_subcategories = sorted(filtered_df['sub_category'].unique())
            selected_subcategories = st.sidebar.multiselect("Sub-Category", all_subcategories, default=[])
            if selected_subcategories:
                filtered_df = filtered_df[filtered_df['sub_category'].isin(selected_subcategories)]
                filtered_raw_df = filtered_raw_df[filtered_raw_df['sub_category'].isin(selected_subcategories)] if filtered_raw_df is not None else None

        if 'brand' in filtered_df.columns:
            filtered_df['brand'] = filtered_df['brand'].fillna('Unknown').astype(str)
            all_brands = sorted(filtered_df['brand'].unique())
            selected_brands = st.sidebar.multiselect("Thương hiệu", all_brands, default=[])
            if selected_brands:
                filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
                filtered_raw_df = filtered_raw_df[filtered_raw_df['brand'].isin(selected_brands)] if filtered_raw_df is not None else None

        if 'category' in filtered_df.columns:
            filtered_df['category'] = filtered_df['category'].fillna('Unknown').astype(str)
            all_categories = sorted(filtered_df['category'].unique())
            if not all_categories:
                st.error("Không có danh mục nào để chọn. Vui lòng kiểm tra dữ liệu đầu vào.")
                return
            normalized_categories = [cat.lower().strip() for cat in all_categories]
            default_categories = ['Kitchen & Dining', 'Home & Kitchen']
            valid_defaults = [cat for cat in default_categories if cat.lower().strip() in normalized_categories]
            if not valid_defaults:
                valid_defaults = all_categories[:2] if len(all_categories) >= 2 else all_categories
            st.sidebar.subheader("Chọn danh mục")
            selected_categories = st.sidebar.multiselect("Danh mục", all_categories, default=valid_defaults)
        else:
            st.error("Dữ liệu không có cột 'category'.")
            return

        if filtered_df.empty:
            st.error("Không có dữ liệu sau khi áp dụng các bộ lọc. Vui lòng điều chỉnh bộ lọc.")
            return

        # Mục phân tích ASIN riêng
        st.header("Phân Tích ASIN", anchor="phân-tích-asin")
        ai = AIInsights()
        asin_input = st.text_input("Nhập ASIN để phân tích (ví dụ: B0ABC12345)", "").strip()
        if asin_input:
            if asin_input in filtered_raw_df['asin'].values:
                st.subheader(f"Phân tích chi tiết ASIN '{asin_input}'")
                asin_analysis = ai.analyze_asin(filtered_df, filtered_raw_df, asin_input)
                for element in asin_analysis:
                    if isinstance(element, str):
                        st.markdown(element)
                    elif isinstance(element, go.Figure):
                        st.plotly_chart(element, use_container_width=True)
                    else:
                        st.write(element)
            else:
                st.error(f"ASIN '{asin_input}' không có trong dữ liệu.")
        st.markdown('<a href="#phân-tích-đối-thủ-amazon" style="float:right;">Back to Top</a>', unsafe_allow_html=True)

        # Các chức năng cũ
        st.header("Dashboard Tổng Quan")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng Doanh Thu", f"${filtered_df['revenue'].sum():,.2f}")
        with col2:
            st.metric("Tổng Số Lượng Mua", f"{filtered_df['bought in past month'].sum():,.0f}")
        with col3:
            st.metric("Giá Trung Bình", f"${filtered_df['price $'].median():,.2f}")

        st.subheader("Danh Sách Sản Phẩm")
        display_cols = ['asin', 'category', 'sub_category', 'brand', 'price $', 'bought in past month', 'revenue']
        if 'image url' in filtered_df.columns:
            display_cols.append('image url')
        st.dataframe(
            filtered_df[display_cols].style.format({
                'price $': '${:.2f}',
                'revenue': '${:.2f}',
                'bought in past month': '{:,.0f}'
            }),
            height=300,
            use_container_width=True
        )

        st.header("Mục Lục")
        st.markdown("- [Phân Tích ASIN](#phân-tích-asin)")
        st.markdown("- [Tổng Quan](#tổng-quan)")
        for cat in selected_categories:
            st.markdown(f"- [Phân Tích Chi Tiết Category '{cat}'](#phân-tích-chi-tiết-category-{cat.replace(' ', '-').replace('&', 'and')})")

        st.header("Tổng Quan", anchor="tổng-quan")
        overview_story, top_subcategories = overview_report_main(filtered_df)
        for element in overview_story:
            if isinstance(element, Paragraph):
                st.markdown(element.getPlainText())
            elif isinstance(element, go.Figure):
                st.plotly_chart(element, use_container_width=True)
            elif isinstance(element, Spacer):
                st.write("")
            else:
                st.warning(f"Không thể hiển thị phần tử: {type(element)}")
        st.markdown('<a href="#phân-tích-đối-thủ-amazon" style="float:right;">Back to Top</a>', unsafe_allow_html=True)

        for category in selected_categories:
            st.header(f"Phân Tích Chi Tiết Category '{category}'", anchor=f"phân-tích-chi-tiết-category-{category.replace(' ', '-').replace('&', 'and')}")
            category_story = category_report_main(filtered_df, top_subcategories, category, selected_categories)
            for element in category_story:
                if isinstance(element, Paragraph):
                    text = element.getPlainText()
                    if text.startswith("•") or text.startswith("Trong đó"):
                        st.markdown(text)
                    else:
                        st.subheader(text)
                elif isinstance(element, go.Figure):
                    st.plotly_chart(element, use_container_width=True)
                elif isinstance(element, Table):
                    for row in element._cellvalues:
                        cols = st.columns(len(row))
                        for idx, cell in enumerate(row):
                            if isinstance(cell, list) and len(cell) == 2:
                                img_buf, caption = cell
                                with cols[idx]:
                                    if isinstance(img_buf, io.BytesIO):
                                        st.image(img_buf.getvalue(), width=150)
                                    else:
                                        st.write("Hình ảnh không khả dụng")
                                    if isinstance(caption, tuple) and len(caption) == 3:
                                        caption_text, url, trend_symbol = caption
                                        st.markdown(f'<a href="{url}" target="_blank">{caption_text}</a> {trend_symbol}', unsafe_allow_html=True)
                                    else:
                                        caption_text = caption.getPlainText() if hasattr(caption, 'getPlainText') else str(caption)
                                        st.caption(caption_text)
                            else:
                                st.write(cell)
                elif isinstance(element, Spacer):
                    st.write("")
                else:
                    st.warning(f"Không thể hiển thị phần tử: {type(element)}")
            st.markdown('<a href="#phân-tích-đối-thủ-amazon" style="float:right;">Back to Top</a>', unsafe_allow_html=True)

        st.sidebar.subheader("Xuất Báo Cáo")
        if st.sidebar.button("Tạo PDF"):
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='VietnameseNormal', fontSize=10, leading=12, spaceAfter=6))
            styles.add(ParagraphStyle(name='VietnameseHeading2', fontSize=14, leading=16, spaceBefore=12, spaceAfter=6))
            styles.add(ParagraphStyle(name='VietnameseTitle', fontSize=16, leading=18, spaceAfter=12))
            styles.add(ParagraphStyle(name='TOCLink', fontSize=10, leading=12, spaceAfter=4))
            styles.add(ParagraphStyle(name='VietnameseCaption', fontSize=8, leading=10, spaceAfter=4, alignment=1))
            styles.add(ParagraphStyle(name='GreenArrow', fontSize=8, textColor=colors.green))
            styles.add(ParagraphStyle(name='RedArrow', fontSize=8, textColor=colors.red))

            toc = [Paragraph("Mục Lục", styles['VietnameseHeading2']), Spacer(1, 12)]
            toc.append(Paragraph('<a href="#overview" color="blue">Tổng Quan</a>', styles['TOCLink']))
            for cat in selected_categories:
                toc.append(Paragraph(f'<a href="#{cat.replace(" ", "_").replace("&", "and")}" color="blue">Phân Tích Chi Tiết Category \'{cat}\'</a>', styles['TOCLink']))

            overview_pdf_story = []
            for element in overview_story:
                if isinstance(element, go.Figure):
                    buf = io.BytesIO()
                    element.write_image(buf, format='png', width=800, height=400)
                    buf.seek(0)
                    overview_pdf_story.append(Image(buf, width=400, height=200))
                else:
                    overview_pdf_story.append(element)
            
            overview_pdf_story.insert(0, Paragraph('<a name="overview"/>', styles['VietnameseNormal']))
            full_story = [Paragraph("Báo Cáo Phân Tích", styles['VietnameseTitle']), Spacer(1, 12)] + toc + overview_pdf_story

            for category in selected_categories:
                category_pdf_story = []
                for element in category_report_main(filtered_df, top_subcategories, category, selected_categories):
                    if isinstance(element, go.Figure):
                        buf = io.BytesIO()
                        element.write_image(buf, format='png', width=800, height=400)
                        buf.seek(0)
                        category_pdf_story.append(Image(buf, width=400, height=200))
                    elif isinstance(element, Table):
                        new_table_data = []
                        for row in element._cellvalues:
                            new_row = []
                            for cell in row:
                                if isinstance(cell, list) and len(cell) == 2:
                                    img_buf, caption = cell
                                    if isinstance(img_buf, io.BytesIO):
                                        img_buf.seek(0)
                                        if isinstance(caption, tuple) and len(caption) == 3:
                                            caption_text, url, trend_symbol = caption
                                            if trend_symbol:
                                                if "green" in trend_symbol:
                                                    arrow = Paragraph("↑", styles['GreenArrow'])
                                                elif "red" in trend_symbol:
                                                    arrow = Paragraph("↓", styles['RedArrow'])
                                                else:
                                                    arrow = Paragraph("", styles['VietnameseCaption'])
                                                caption_for_pdf = [
                                                    Paragraph(f'<link href="{url}" color="blue">{caption_text}</link>', styles['VietnameseCaption']),
                                                    arrow
                                                ]
                                            else:
                                                caption_for_pdf = Paragraph(f'<link href="{url}" color="blue">{caption_text}</link>', styles['VietnameseCaption'])
                                            new_row.append([Image(img_buf, width=100, height=100), caption_for_pdf])
                                        else:
                                            new_row.append([Image(img_buf, width=100, height=100), caption])
                                    else:
                                        new_row.append(cell)
                                else:
                                    new_row.append(cell)
                            new_table_data.append(new_row)
                        category_pdf_story.append(Table(new_table_data, colWidths=[150] * len(new_table_data[0])))
                    else:
                        category_pdf_story.append(element)
                full_story.extend(category_pdf_story)

            pdf_path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            doc.build(full_story)
            st.sidebar.markdown(get_binary_file_downloader_html(pdf_path, "Tải Báo Cáo PDF"))

    finally:
        # Xóa thư mục tạm sau khi xử lý
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)

if __name__ == "__main__":
    main()
