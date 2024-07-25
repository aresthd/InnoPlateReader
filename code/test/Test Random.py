import os
import shutil
from datetime import datetime, timedelta

def create_dir(base_dir):
    # Tanggal hari ini
    today = datetime.now()
    today_dir_name = today.strftime('%d-%m-%Y')
    today_dir_path = os.path.join(base_dir, today_dir_name)
    
    # Tanggal 10 hari yang lalu
    ten_days_ago = today - timedelta(days=10)

    # Cek semua direktori di base_dir
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        
        if os.path.isdir(dir_path):
            try:
                dir_date = datetime.strptime(dir_name, '%d-%m-%Y')
                # Hapus direktori yang lebih dari 10 hari yang lalu
                if dir_date < ten_days_ago:
                    shutil.rmtree(dir_path)
                    print(f"Direktori {dir_name} telah dihapus.")
            except ValueError:
                # Jika nama direktori tidak sesuai format tanggal, lewati
                continue
    
    # Buat direktori untuk hari ini jika belum ada
    if not os.path.exists(today_dir_path):
        os.makedirs(today_dir_path)
        print(f"Direktori {today_dir_name} telah dibuat.")
    else:
        print(f"Direktori {today_dir_name} sudah ada.")

    return today_dir_name

basedirTraining = os.path.join(os.getcwd(), 'static', 'training')
print(f'Before : {basedirTraining}')

basedirTraining = os.path.join('static', 'training', create_dir(basedirTraining))
print(f'After : {basedirTraining}')
