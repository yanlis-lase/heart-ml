Opsi 1 — Install Python via Windows Package Manager (winget) ✅ REKOMENDASI
Kalau Windows kamu Windows 10/11 modern, biasanya winget sudah ada.

1️⃣ Cek dulu
Di PowerShell (VS Code terminal juga boleh):
	winget --version
Kalau keluar versi → lanjut.

2️⃣ Install Python (resmi, aman)
	winget install -e --id Python.Python.3.11
Ini akan:
install Python resmi dari python.org
otomatis set PATH (lebih rapi dari Microsoft Store)
Tunggu sampai selesai.

3️⃣ Tutup terminal → buka VS Code
Lalu cek:
	python --version
atau
	py -V
Kalau keluar versi → beres.

3️⃣ Setelah Python terpasang
Begitu python --version sudah berhasil, lakukan berurutan:
	masuk ke root folder, klik kanan, 'Open In Terminal', jendela powerShell akan terbuka
	py -m venv venv
	.\venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	
3️⃣ run script dengan format seperti ini	
python .\src\baseline.py
