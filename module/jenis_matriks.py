import numpy as np

def matriks_nol():
    baris_nol = int(input("Masukkan jumlah baris matriks nol: "))
    kolom_nol = int(input("Masukkan jumlah kolom matriks nol: "))
    matriks = np.zeros((baris_nol, kolom_nol))
    print("\nMatriks Nol:")
    print(matriks)

def matriks_bujur_sangkar():
    ukuran_bujur_sangkar = int(input("\nMasukkan ukuran matriks bujur sangkar: "))
    matriks = np.random.randint(9, size=(ukuran_bujur_sangkar, ukuran_bujur_sangkar))
    print("\nMatriks Bujur Sangkar:")
    print(matriks)

def matriks_persegi_panjang():
    baris_persegi_panjang = int(input("\nMasukkan jumlah baris matriks persegi panjang: "))
    kolom_persegi_panjang = int(input("Masukkan jumlah kolom matriks persegi panjang: "))
    matriks = np.random.randint(9, size=(baris_persegi_panjang, kolom_persegi_panjang))
    print("\nMatriks Persegi Panjang:")
    print(matriks)

def matriks_diagonal():
    ukuran_diagonal = int(input("\nMasukkan ukuran matriks diagonal: "))
    elemen_diagonal = np.random.randint(1, 10, ukuran_diagonal)
    matriks = np.diag(elemen_diagonal)
    print("\nMatriks Diagonal:")
    print(matriks)

def matriks_identitas():
    ukuran_identitas = int(input("\nMasukkan ukuran matriks identitas: "))
    matriks = np.eye(ukuran_identitas)
    print("\nMatriks Identitas:")
    print(matriks)

def matriks_skalar():
    ukuran_skalar = int(input("\nMasukkan ukuran matriks skalar: "))
    nilai_skalar = int(input("Masukkan nilai skalar: "))
    matriks = nilai_skalar * np.eye(ukuran_skalar)
    print("\nMatriks Skalar:")
    print(matriks)

def matriks_menu():
    while True:
        print("\nMenu Jenis Matriks")
        print("1. Matriks Nol")
        print("2. Matriks Bujur Sangkar")
        print("3. Matriks Persegi Panjang")
        print("4. Matriks Diagonal")
        print("5. Matriks Identitas")
        print("6. Matriks Skalar")
        print("0. Kembali ke Menu Utama")
        pilihan_matriks = input("Pilih menu: ")

        if pilihan_matriks == "1":
            matriks_nol()
        elif pilihan_matriks == "2":
            matriks_bujur_sangkar()
        elif pilihan_matriks == "3":
            matriks_persegi_panjang()
        elif pilihan_matriks == "4":
            matriks_diagonal()
        elif pilihan_matriks == "5":
            matriks_identitas()
        elif pilihan_matriks == "6":
            matriks_skalar()
        elif pilihan_matriks == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")