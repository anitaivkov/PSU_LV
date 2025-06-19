import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 

###3. zadatak
def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y, noise_level=0.1):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + noise_level * varNoise * np.random.normal(0,1,len(y))
    return y_noisy

degrees = [2, 6, 15]
mse_train_results = []
mse_test_results = []

num_samples_original = 100
num_samples_large = 500
num_samples_small = 20

def run_simulation(num_samples, title_suffix=""):
    print(f"\n--- Simulacija sa {num_samples} uzoraka {title_suffix} ---")
    x = np.linspace(1, 10, num_samples)
    y_true = non_func(x)
    y_measured = add_noise(y_true)

    x = x[:, np.newaxis]
    y_measured = y_measured[:, np.newaxis]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y_measured, test_size=0.3, random_state=12)

    fig, axes = plt.subplots(1, len(degrees), figsize=(18, 6), sharey=True)
    fig.suptitle(f'Usporedba modela s različitim stupnjevima za {num_samples} uzoraka {title_suffix}')

    current_mse_train = []
    current_mse_test = []

    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree)
        xtrain_poly = poly.fit_transform(xtrain)
        xtest_poly = poly.transform(xtest)

        linear_model = lm.LinearRegression()
        linear_model.fit(xtrain_poly, ytrain)

        ypred_train = linear_model.predict(xtrain_poly)
        ypred_test = linear_model.predict(xtest_poly)

        mse_train = mean_squared_error(ytrain, ypred_train)
        mse_test = mean_squared_error(ytest, ypred_test)

        current_mse_train.append(mse_train)
        current_mse_test.append(mse_test)

        print(f"Degree {degree}: MSE_train = {mse_train:.4f}, MSE_test = {mse_test:.4f}")

        ax = axes[i] if len(degrees) > 1 else axes
        sort_idx = np.argsort(x[:,0])
        x_sorted = x[sort_idx]
        y_true_sorted = y_true[sort_idx]
        
        x_full_poly = poly.transform(x_sorted)
        ypred_full = linear_model.predict(x_full_poly)


        ax.plot(x_sorted, y_true_sorted, 'k--', label='Stvarna funkcija')
        ax.plot(xtest[:,0], ytest, 'o', color='blue', alpha=0.6, label='Test podaci')
        ax.plot(xtrain[:,0], ytrain, 'x', color='red', alpha=0.6, label='Trening podaci')
        ax.plot(x_sorted, ypred_full, '-', color='green', label=f'Model (deg={degree})')
        ax.set_title(f'Degree = {degree}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'model_comparison_{num_samples_original}_samples_{title_suffix}.png')
    plt.show()

    return current_mse_train, current_mse_test


mse_train_original, mse_test_original = run_simulation(num_samples_original, "originalni")
print(f"\nMSE trening rezultati (original): {mse_train_original}")
print(f"MSE test rezultati (original): {mse_test_original}")

mse_train_large, mse_test_large = run_simulation(num_samples_large, "veci broj uzoraka")
print(f"\nMSE trening rezultati (veci broj uzoraka): {mse_train_large}")
print(f"MSE test rezultati (veci broj uzoraka): {mse_test_large}")

mse_train_small, mse_test_small = run_simulation(num_samples_small, "manji broj uzoraka")
print(f"\nMSE trening rezultati (manji broj uzoraka): {mse_train_small}")
print(f"MSE test rezultati (manji broj uzoraka): {mse_test_small}")
