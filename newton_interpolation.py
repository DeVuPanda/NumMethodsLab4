import numpy as np
import matplotlib.pyplot as plt

def divided_difference(x, y):

    n = len(x)
    table = np.zeros((n, n))

    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table

def print_divided_difference_table(x, table):
    n = len(x)

    col_width = 12

    print("\nТаблиця розділених різниць:")
    print("-" * (col_width * (n + 1)))

    header = f"{'x'.center(col_width)}{'f(x)'.center(col_width)}"
    for i in range(1, n):
        header += f"f[{i}]".center(col_width)
    print(header)
    print("-" * (col_width * (n + 1)))

    for i in range(n):
        row = f"{x[i]:.3f}".center(col_width) + f"{table[i][0]:.6f}".center(col_width)
        for j in range(1, n - i):
            row += f"{table[i][j]:.6f}".center(col_width)
        print(row)

    print("-" * (col_width * (n + 1)))

def newton_polynomial(coef, x_data, x):

    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

def get_newton_polynomial_str(x_data, coef):

    n = len(coef)
    poly_str = f"{coef[0]:.6f}"

    for i in range(1, n):
        if coef[i] >= 0:
            poly_str += " + "
        else:
            poly_str += " - "

        term = f"{abs(coef[i]):.6f}"
        for j in range(i):
            term += f"(x - {x_data[j]:.3f})"

        poly_str += term

    formatted_str = ""
    line_length = 0

    for char in poly_str:
        formatted_str += char
        line_length += 1
        if line_length >= 150 and char == " ":
            formatted_str += "\n"
            line_length = 0

    return formatted_str

x = np.array([0, 0.15, 0.35, 0.6, 0.9, 1.1, 1.4, 1.57, 1.8, 2, 2.3, 2.6, 2.8, 3, 3.1])
y = np.array([1, 0.988, 0.94, 0.825, 0.621, 0.453, 0.169, 0, -0.227, -0.416, -0.669, -0.857, -0.942, -0.99, -1])

div_diff_table = divided_difference(x, y)

print_divided_difference_table(x, div_diff_table)

coefficients = [div_diff_table[0][j] for j in range(len(x))]

print("\nПоліном:")
print(get_newton_polynomial_str(x, coefficients))

x_range = np.linspace(0, np.pi, 500)

y_newton = [newton_polynomial(coefficients, x, xi) for xi in x_range]

y_cos = np.cos(x_range)

plt.figure(figsize=(12, 7))
plt.plot(x_range, y_newton, label='Інтерполяційний поліном Ньютона', color='blue', linewidth=2)
plt.plot(x_range, y_cos, label='cos(x)', color='red', linestyle='--', linewidth=2)

plt.title('Порівняння f(x) = cos(x) та інтерполяційного поліному Ньютона', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



