import requests
from tqdm import tqdm
import time

def download(url: str, fname: str):
    retries = 5  # 设置重试次数
    for attempt in range(retries):
        try:
            # 用流stream的方式获取url的数据
            resp = requests.get(url, stream=True)
            # 拿到文件的长度，并把total初始化为0
            total = int(resp.headers.get('content-length', 0))
            # 打开当前目录的fname文件(名字你来传入)
            # 初始化tqdm，传入总数，文件名等数据，接着就是写入，更新等操作了
            with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            break  # 如果下载成功，跳出重试循环
        except Exception as e:
            print(f"Error occurred during download: {e}")
            if attempt < retries - 1:
                print(f"Retrying ({attempt + 1}/{retries})...")
                time.sleep(5)  # 间隔5秒后重试
            else:
                print("Max retries reached. Download failed.")
                return

monthday = [0,31,28,31,30,31,30,31,31,30,31,30,31]

for year in range(2000,2010):
    for month in range(1,13):
        url1 = r'https://data-cbr.csiro.au/thredds/ncss/catch_all/CMAR_CAWCR-Wave_archive/CAWCR_Wave_Hindcast_aggregate/gridded/ww3.aus_4m.'\
               + str(year) + str(month).zfill(2) + \
               r'.nc?var=U10&var=V10&var=dir&var=hs&var=t&var=t02&var=tm0m1&north=0&west=135&east=170&south=-50.0000&disableProjSubset=on&horizStride=1&time_start='\
               + str(year) + '-' + str(month).zfill(2) + r'-01T00%3A00%3A00Z&time_end=' + str(year) + '-' + str(month).zfill(2) + '-' + str(monthday[month]) + \
               r'T23%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf4'
        download(url1, str(year)+str(month).zfill(2)+'hiRes.nc4')
        
        url2 = r'https://data-cbr.csiro.au/thredds/ncss/catch_all/CMAR_CAWCR-Wave_archive/CAWCR_Wave_Hindcast_aggregate/gridded/ww3.glob_24m.'\
               + str(year) + str(month).zfill(2) + \
               r'.nc?var=dir&var=hs&var=t02&var=t0m1&var=uwnd&var=vwnd&&north=-35&west=137&east=155&south=-45&disableProjSubset=on&horizStride=1&time_start='\
               + str(year) + '-' + str(month).zfill(2) + r'-01T00%3A00%3A00Z&time_end=' + str(year) + '-' + str(month).zfill(2) + '-' + str(monthday[month]) + \
               r'T23%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf4'
        print(url2)
        download(url2, str(year)+str(month).zfill(2)+'lowRes.nc4')


