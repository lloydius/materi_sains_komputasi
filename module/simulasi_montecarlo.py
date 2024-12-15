import numpy as np
import pandas as pd

def distribusi_permintaan():

    n = int(input("Masukkan jumlah permintaan: "))
    permintaan = []
    probabilitas = []

    print("Masukkan nilai permintaan dan probabilitasnya (dalam format: permintaan probabilitas):")
    for _ in range(n):
        p, prob = map(float, input().split())
        permintaan.append(int(p))
        probabilitas.append(prob)

    if not np.isclose(sum(probabilitas), 1):
        print("Error: Probabilitas harus berjumlah 1.")
        return

    probabilitas_kumulatif = np.cumsum(probabilitas)

    # Tentukan interval berdasarkan probabilitas kumulatif
    interval = []
    for i in range(len(probabilitas_kumulatif)):
        awal = int(probabilitas_kumulatif[i-1] * 100) + 1 if i > 0 else 1
        akhir = int(probabilitas_kumulatif[i] * 100)
        interval.append((awal, akhir))

    # Tampilkan tabel interval bilangan acak
    print("\nTabel Interval Bilangan Acak:")
    for i, (perm, inter) in enumerate(zip(permintaan, interval)):
        print(f"Permintaan: {perm} | Interval: {inter[0]:02d} - {inter[1]:02d}")

    n_hari = int(input("\nMasukkan jumlah hari untuk simulasi: "))

    bilangan_acak = np.random.randint(1, 101, size=n_hari)

    # Tentukan permintaan berdasarkan bilangan acak
    hasil_simulasi = []
    for bilangan in bilangan_acak:
        for i, inter in enumerate(interval):
            if inter[0] <= bilangan <= inter[1]:
                hasil_simulasi.append(permintaan[i])
                break

    simulasi_df = pd.DataFrame({
        "Hari": range(1, n_hari + 1),
        "Bilangan Acak": bilangan_acak,
        "Permintaan": hasil_simulasi
    })

    print("\nTabel Simulasi:")
    print(simulasi_df)

    rata_rata = np.mean(hasil_simulasi)
    print(f"\nRata-rata permintaan per hari: {rata_rata:.2f}")

    ekspektasi = sum([p * d for p, d in zip(probabilitas, permintaan)])
    print(f"Ekspektasi (E) permintaan: {ekspektasi:.2f}")

def monte_carlo_menu():
    while True:
        print("\nMenu Monte Carlo Integration")
        print("1. Distribusi Permintaan")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            distribusi_permintaan()
        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")