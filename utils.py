import aiohttp
import io
import asyncio
from PIL import Image as PILImage
import pandas as pd

# Hàm tải hình ảnh bất đồng bộ
async def download_image_async(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                response.raise_for_status()
                content = await response.read()
                image = PILImage.open(io.BytesIO(content))
                image.thumbnail((100, 100))
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                buf.seek(0)
                return buf
    except Exception as e:
        print(f"Lỗi khi tải hình ảnh từ {url}: {str(e)}")
        return None

# Hàm tải nhiều hình ảnh bất đồng bộ
async def download_images(urls):
    tasks = [download_image_async(url) for url in urls]
    return await asyncio.gather(*tasks)

# Hàm tính xu hướng giá
def get_price_trend(asin, df):
    if 'date' not in df.columns:
        return None
    asin_data = df[df['asin'] == asin].sort_values(by='date', ascending=False)
    if len(asin_data) < 2:
        return None
    current_price = asin_data.iloc[0]['price $']
    previous_price = asin_data.iloc[1]['price $']
    if current_price > previous_price:
        return "up"
    elif current_price < previous_price:
        return "down"
    return None