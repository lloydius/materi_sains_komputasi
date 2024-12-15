import numpy as np

def jacobi_iteration(x0, max_iterations, tolerance):
    global x_new

    def x1_new(x2, x3):
        return (1 / 10) * x2 - (2 / 10) * x3 + (6 / 10)

    def x2_new(x1, x3, x4):
        return (1 / 11) * x1 + (1 / 11) * x3 - (3 / 11) * x4 + (25 / 11)

    def x3_new(x1, x2, x4):
        return -(2 / 10) * x1 + (1 / 10) * x2 + (1 / 10) * x4 - (11 / 10)

    def x4_new(x2, x3):
        return -(3 / 8) * x2 + (1 / 8) * x3 + (15 / 8)

    x_old = np.array(x0, dtype=float)
    prev_norm_l2 = None
    prev_norm_inf = None

    print(f"Nilai awal: x = {x_old}")

    for k in range(1, max_iterations + 1):
        x_new = np.zeros_like(x_old)
        x_new[0] = x1_new(x_old[1], x_old[2])
        x_new[1] = x2_new(x_old[0], x_old[2], x_old[3])
        x_new[2] = x3_new(x_old[0], x_old[1], x_old[3])
        x_new[3] = x4_new(x_old[1], x_old[2])

        norm_l2 = np.sqrt(np.sum(x_new**2))
        norm_inf = np.max(np.abs(x_new))

        epsilon_l2 = abs(norm_l2 - prev_norm_l2) if prev_norm_l2 is not None else None
        epsilon_inf = abs(norm_inf - prev_norm_inf) if prev_norm_inf is not None else None

        print(f"Iterasi {k}: x = {x_new}, norm_l2 = {norm_l2:.4f}, norm_inf = {norm_inf:.4f}", end="")
        if epsilon_l2 is not None and epsilon_inf is not None:
            print(f", epsilon_l2 = {epsilon_l2:.4f}, epsilon_inf = {epsilon_inf:.4f}")
        else:
            print()

        if epsilon_l2 is not None and epsilon_inf is not None and max(epsilon_l2, epsilon_inf) < tolerance:
            print("\nSolusi konvergen:")
            print(x_new)
            print(f"Konvergensi tercapai dalam {k} iterasi.")
            return x_new

        x_old = x_new
        prev_norm_l2 = norm_l2
        prev_norm_inf = norm_inf

    print("Iterasi maksimum tercapai tanpa konvergensi.")
    return x_new

def gauss_seidel_iteration(x0, max_iterations, tolerance):
    global x_new

    def x1_new(x2, x3):
        return (1 / 10) * x2 - (2 / 10) * x3 + (6 / 10)

    def x2_new(x1, x3, x4):
        return (1 / 11) * x1 + (1 / 11) * x3 - (3 / 11) * x4 + (25 / 11)

    def x3_new(x1, x2, x4):
        return -(2 / 10) * x1 + (1 / 10) * x2 + (1 / 10) * x4 - (11 / 10)

    def x4_new(x2, x3):
        return -(3 / 8) * x2 + (1 / 8) * x3 + (15 / 8)

    x_old = np.array(x0, dtype=float)
    prev_norm_l2 = None
    prev_norm_inf = None

    print(f"Nilai awal: x = {x_old}")

    for k in range(1, max_iterations + 1):
        x_new = x_old.copy()
        x_new[0] = x1_new(x_new[1], x_new[2])
        x_new[1] = x2_new(x_new[0], x_new[2], x_new[3])
        x_new[2] = x3_new(x_new[0], x_new[1], x_new[3])
        x_new[3] = x4_new(x_new[1], x_new[2])

        norm_l2 = np.sqrt(np.sum(x_new**2))
        norm_inf = np.max(np.abs(x_new))

        epsilon_l2 = abs(norm_l2 - prev_norm_l2) if prev_norm_l2 is not None else None
        epsilon_inf = abs(norm_inf - prev_norm_inf) if prev_norm_inf is not None else None

        print(f"Iterasi {k}: x = {x_new}, norm_l2 = {norm_l2:.4f}, norm_inf = {norm_inf:.4f}", end="")
        if epsilon_l2 is not None and epsilon_inf is not None:
            print(f", epsilon_l2 = {epsilon_l2:.4f}, epsilon_inf = {epsilon_inf:.4f}")
        else:
            print()

        if epsilon_l2 is not None and epsilon_inf is not None and max(epsilon_l2, epsilon_inf) < tolerance:
            print("\nSolusi konvergen:")
            print(x_new)
            print(f"Konvergensi tercapai dalam {k} iterasi.")
            return x_new

        x_old = x_new
        prev_norm_l2 = norm_l2
        prev_norm_inf = norm_inf

    print("Iterasi maksimum tercapai tanpa konvergensi.")
    return x_new

def iterasi_menu():
    while True:
        print("\nMenu Iterasi")
        print("1. Iterasi Jacobi")
        print("2. Iterasi Gauss-Seidel")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih metode iterasi: ")

        if pilihan == "1" or pilihan == "2":
            x0 = list(map(float, input("Masukkan nilai awal x (dipisahkan dengan spasi): ").split()))
            max_iterations = int(input("Masukkan jumlah iterasi maksimum: "))
            tolerance = float(input("Masukkan toleransi: "))

            if pilihan == "1":
                print("\nIterasi Jacobi:")
                jacobi_iteration(x0, max_iterations, tolerance)
            else:
                print("\nIterasi Gauss-Seidel:")
                gauss_seidel_iteration(x0, max_iterations, tolerance)

        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")