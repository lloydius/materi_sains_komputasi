import numpy as np
import sympy as sp
import math
import pandas as pd
import sys
sys.path.append("module")

from jenis_matriks import *
from operasi_matriks import *
from iterasi import *
from interpolasi import *
from pers_nonlinier import *
from simulasi_montecarlo import *

def main_menu():
    while True:
        print("\nMenu Utama")
        print("1. Jenis Matriks")
        print("2. Operasi Matriks")
        print("3. Interpolasi")
        print("4. Iterasi")
        print("5. Simulasi Monte Carlo")
        print("0. Keluar")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            matriks_menu()
        elif pilihan == "2":
            matriks_operasi_menu()
        elif pilihan == "3":
            interpolasi_menu()
        elif pilihan == "4":
            iterasi_menu()
        elif pilihan == "5":
            monte_carlo_menu()
        elif pilihan == "6":
            pers_non_linier_menu()
        elif pilihan == "0":
            print("Keluar dari program.")
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

if __name__ == "__main__":
    main_menu()
