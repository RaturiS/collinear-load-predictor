# RF Collinear Load — Machine Learning Surrogate Model Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, max_error
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.size']        = 11
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['axes.labelsize']   = 11
plt.rcParams['legend.fontsize']  = 9
plt.rcParams['figure.dpi']       = 110


df = pd.read_csv('data_rewr.csv')
df.columns = df.columns.str.strip()

df = df.rename(columns={
    'cell_radius':          'Radius_mm',
    'length':               'Coating_mm',
    'freq (2pi_by_3 mode)': 'Frequency_MHz',
    'Quality factor':       'Q_factor'
})
df = df[['Radius_mm', 'Coating_mm', 'Frequency_MHz', 'Q_factor']].dropna()

print(f"Loaded {len(df)} simulation rows.")
print(f"Unique radii: {sorted(df['Radius_mm'].unique())}\n")

# Colour palette — one colour per unique radius value
unique_radii = sorted(df['Radius_mm'].unique())
colors       = cm.plasma(np.linspace(0.1, 0.85, len(unique_radii)))
color_map    = {r: colors[i] for i, r in enumerate(unique_radii)}


# EXPLORATORY DATA ANALYSIS (EDA)

print("Generating Exploratory Figures...")


fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['Coating_mm'], df['Frequency_MHz'], color='steelblue', s=55, alpha=0.75, edgecolors='white', linewidth=0.5)
ax.axhline(2856, color='red', linewidth=2, linestyle='--', label='Target: 2856 MHz')
ax.set_xlabel('Coating Length (mm)')
ax.set_ylabel('Resonant Frequency (MHz)')
ax.set_title('Frequency vs Coating Length\n')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('freq_raw.png', dpi=180, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(7, 5))
for r in unique_radii:
    subset = df[df['Radius_mm'] == r].sort_values('Coating_mm')
    ax.plot(subset['Coating_mm'], subset['Frequency_MHz'], 'o-', color=color_map[r], markersize=5, linewidth=1.6, label=f'R = {r:.4f} mm')
ax.axhline(2856, color='red', linewidth=2, linestyle='--', label='Target: 2856 MHz')
ax.set_xlabel('Coating Length (mm)')
ax.set_ylabel('Resonant Frequency (MHz)')
ax.set_title('Frequency vs Coating Length\n(Coloured by Cell Radius)')
ax.legend(fontsize=7.5, loc='lower left', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('freq_colored.png', dpi=180, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(7, 5))
for r in unique_radii:
    subset = df[df['Radius_mm'] == r].sort_values('Coating_mm')
    ln_L   = np.log(subset['Coating_mm'].values)
    ln_Q   = np.log(subset['Q_factor'].values)
    ax.scatter(ln_L, ln_Q, color=color_map[r], s=45, edgecolors='white', linewidth=0.4, label=f'R = {r:.4f} mm', zorder=4)

ln_L_all = np.log(df['Coating_mm'].values)
ln_Q_all = np.log(df['Q_factor'].values)
coeffs   = np.polyfit(ln_L_all, ln_Q_all, 1)          
r2_fit   = r2_score(ln_Q_all, np.polyval(coeffs, ln_L_all))
ln_L_fit = np.linspace(ln_L_all.min(), ln_L_all.max(), 200)
ln_Q_fit = np.polyval(coeffs, ln_L_fit)

ax.plot(ln_L_fit, ln_Q_fit, 'r-', linewidth=2.2, zorder=5, label=f'Global fit:  slope = {coeffs[0]:.4f}\n$R^2$ = {r2_fit:.5f}')
ax.set_xlabel('ln (Coating Length)')
ax.set_ylabel('ln (Quality Factor $Q$)')
ax.set_title('�Log Plot: Power-Law Confirmation\n' r'$\ln(Q) = n\,\ln(L) + c$')
ax.legend(fontsize=7.5, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('log_q.png', dpi=180, bbox_inches='tight')
plt.close()

# MODEL TRAINING & PIPELINE

print("Training Log-Polynomial Model")

X_all  = df[['Q_factor', 'Frequency_MHz']].values
y_len  = df['Coating_mm'].values
y_rad  = df['Radius_mm'].values

X_train, X_test, yL_train, yL_test, yR_train, yR_test = train_test_split(
    X_all, y_len, y_rad, test_size=0.2, random_state=42
)

def log_q_transform(data):
    d = np.array(data, dtype=float)
    d[:, 0] = np.log(d[:, 0])
    return d

length_predictor = make_pipeline(
    FunctionTransformer(log_q_transform),
    PolynomialFeatures(degree=3),
    LinearRegression()
)

radius_predictor = make_pipeline(
    FunctionTransformer(log_q_transform),
    PolynomialFeatures(degree=3),
    LinearRegression()
)

length_predictor.fit(X_train, yL_train)
radius_predictor.fit(X_train, yR_train)

L_pred = length_predictor.predict(X_test)
R_pred = radius_predictor.predict(X_test)

r2_L = r2_score(yL_test, L_pred)
r2_R = r2_score(yR_test, R_pred)


# MODEL EVALUATION FIGURES ===

print("Generating Evaluation Fig")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Coating Length

ax1.scatter(yL_test, L_pred, color='steelblue', s=80, edgecolors='white', linewidth=0.6, zorder=4, label='Test set predictions')
lim1 = [min(yL_test.min(), L_pred.min()) - 0.5, max(yL_test.max(), L_pred.max()) + 0.5]
ax1.plot(lim1, lim1, 'k--', linewidth=1.8, label='Perfect 1:1 fit')
ax1.set_xlabel('CST Simulated Coating Length (mm)')
ax1.set_ylabel('Predicted Coating Length (mm)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(lim1)
ax1.set_ylim(lim1)
ax1.set_aspect('equal')


ax2.scatter(yR_test, R_pred, color='darkorange', s=80, edgecolors='white', linewidth=0.6, zorder=4, label='Test set predictions')
lim2 = [min(yR_test.min(), R_pred.min()) - 0.005, max(yR_test.max(), R_pred.max()) + 0.005]
ax2.plot(lim2, lim2, 'k--', linewidth=1.8, label='Perfect 1:1 fit')
ax2.set_xlabel('CST Simulated Cell Radius (mm)')
ax2.set_ylabel('Predicted Cell Radius (mm)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(lim2)
ax2.set_ylim(lim2)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=180, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

residuals_L = L_pred - yL_test
ax1.scatter(yL_test, residuals_L, color='purple', s=60, alpha=0.7, edgecolor='white', zorder=4)
ax1.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax1.set_xlabel('True CST Simulated Coating Length (mm)')
ax1.set_ylabel('Prediction Error in Coating Length (mm)')
ax1.grid(True, alpha=0.3)

residuals_R = R_pred - yR_test
ax2.scatter(yR_test, residuals_R, color='blue', s=60, alpha=0.7, edgecolor='white', zorder=4)
ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax2.set_xlabel('True CST Simulated Radius (mm)')
ax2.set_ylabel('Prediction Error in Radius (mm)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_error_combined.png', dpi=180, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(7, 5))
train_sizes, train_scores, test_scores = learning_curve(
    length_predictor, X_all, y_len, cv=5, 
    train_sizes=np.linspace(0.2, 1.0, 10), scoring='r2', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score ($R^2$)')
ax.plot(train_sizes, test_mean, 's-', color='green', label='Cross-Validation Score ($R^2$)')
ax.set_xlabel('Number of CST Training Simulations')
ax.set_ylabel('Model Accuracy ($R^2$)')
ax.set_title('Learning Curve\n(Data Efficiency)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=180, bbox_inches='tight')
plt.close()


#  METRICS 

mae_L = mean_absolute_error(yL_test, L_pred)
mae_R = mean_absolute_error(yR_test, R_pred)
max_err_L = max_error(yL_test, L_pred)
max_err_R = max_error(yR_test, R_pred)

print("\n" + "="*60)
print(" TABLE 2: Summary of model performance on 20% test data")
print("="*60)
print(f"{'Metric':<25} | {'Coating Length':<15} | {'Cell Radius'}")
print("-" * 60)
print(f"{'R2 score (test set)':<25} | {r2_L:<15.4f} | {r2_R:.4f}")
print(f"{'Mean Absolute Error':<25} | {mae_L:<12.4f} mm | {mae_R:.6f} mm")
print(f"{'Maximum observed error':<25} | {max_err_L:<12.4f} mm | {max_err_R:.6f} mm")
print("=" * 60 + "\n")


#  MATHEMATICAL weighrts  

poly_tool = length_predictor.named_steps['polynomialfeatures']
feature_names = poly_tool.get_feature_names_out(['ln(Q)', 'Freq'])
length_model = length_predictor.named_steps['linearregression']
radius_model = radius_predictor.named_steps['linearregression']

print("\n" + "="*55)
print(" MATHEMATICAL EQUATION WEIGHTS: COATING LENGTH (L)")
print("="*55)
print(f"Intercept (Base Value) : {length_model.intercept_:+.6f}\n")
for name, weight in zip(feature_names, length_model.coef_):
    if name != "1": 
        print(f"Weight for [ {name:<15} ] : {weight:+.6f}")

print("\n" + "="*55)
print(" MATHEMATICAL EQUATION WEIGHTS: CELL RADIUS (R)")
print("="*55)
print(f"Intercept (Base Value) : {radius_model.intercept_:+.6f}\n")
for name, weight in zip(feature_names, radius_model.coef_):
    if name != "1":
        print(f"Weight for [ {name:<15} ] : {weight:+.6f}")
print("="*55 + "\n")


print("Generating 3D Response Surface plots...")
fig = plt.figure(figsize=(14, 6))

q_range = np.linspace(df['Q_factor'].min(), df['Q_factor'].max(), 50)
f_range = np.linspace(df['Frequency_MHz'].min(), df['Frequency_MHz'].max(), 50)
Q_mesh, F_mesh = np.meshgrid(q_range, f_range)
grid_inputs = np.c_[Q_mesh.ravel(), F_mesh.ravel()]

L_surface = length_predictor.predict(grid_inputs).reshape(Q_mesh.shape)
R_surface = radius_predictor.predict(grid_inputs).reshape(Q_mesh.shape)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Q_mesh, F_mesh, L_surface, cmap='viridis', alpha=0.6, edgecolor='none')
ax1.scatter(df['Q_factor'], df['Frequency_MHz'], df['Coating_mm'], color='red', s=30, edgecolor='black', label='Actual CST Data', zorder=5)
ax1.set_xlabel('\nQ Factor')
ax1.set_ylabel('\nFrequency (MHz)')
ax1.set_zlabel('\nCoating Length (mm)')
ax1.set_title('Coating Length Prediction Surface\n(The 3rd-Order Polynomial)')
ax1.view_init(elev=20, azim=-135) 
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Q_mesh, F_mesh, R_surface, cmap='plasma', alpha=0.6, edgecolor='none')
ax2.scatter(df['Q_factor'], df['Frequency_MHz'], df['Radius_mm'], color='black', s=30, edgecolor='white', label='Actual CST Data', zorder=5)
ax2.set_xlabel('\nQ Factor')
ax2.set_ylabel('\nFrequency (MHz)')
ax2.set_zlabel('\nCell Radius (mm)')
ax2.set_title('Cell Radius Prediction Surface\n(The 3rd-Order Polynomial)')
ax2.view_init(elev=20, azim=-135)
ax2.legend()
plt.tight_layout()
plt.savefig('3d_response_surface.png', dpi=180, bbox_inches='tight')
plt.close()


print("Generating the Error Bowl plot...")
Z_train = length_predictor.named_steps['polynomialfeatures'].transform(
    length_predictor.named_steps['functiontransformer'].transform(X_train)
)
y_true = yL_train
w_opt = length_model.coef_.copy()      
intercept_opt = length_model.intercept_ 

w0_guesses = np.linspace(intercept_opt - 10, intercept_opt + 10, 40)
w1_guesses = np.linspace(w_opt[1] - 5, w_opt[1] + 5, 40)
W0, W1 = np.meshgrid(w0_guesses, w1_guesses)
MSE_surface = np.zeros_like(W0)

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w_temp = w_opt.copy()
        w_temp[1] = W1[i, j] 
        b_temp = W0[i, j]    
        y_guess = np.dot(Z_train, w_temp) + b_temp
        mse = np.mean((y_true - y_guess)**2)
        MSE_surface[i, j] = mse

y_perfect = np.dot(Z_train, w_opt) + intercept_opt
min_mse = np.mean((y_true - y_perfect)**2)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, MSE_surface, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax.scatter([intercept_opt], [w_opt[1]], [min_mse], color='lime', s=300, marker='*', edgecolor='black', linewidth=1.5, label='Global Minimum\n(Found via Normal Equation)', zorder=10)
ax.set_xlabel('\nIntercept Weight Guess ($w_0$)')
ax.set_ylabel('\n$\ln(Q)$ Weight Guess ($w_1$)')
ax.set_zlabel('\nMean Squared Error (Loss)')
ax.set_title('The Convex Error Bowl (Cost Function)\nVisualizing the Mathematical Optimization')
ax.view_init(elev=25, azim=-125)
ax.legend()
plt.tight_layout()
plt.savefig('error_bowl.png', dpi=180, bbox_inches='tight')
plt.close()


print("Generating Computational Speedup plot...")
fig, ax = plt.subplots(figsize=(7, 4))
cst_time_seconds = 15 * 60  
ml_time_seconds = 0.002     

methods = ['CST MWS Simulation\n(Full 3D Eigenmode)', 'ML Surrogate Model\n(Log-Polynomial)']
times = [cst_time_seconds, ml_time_seconds]
bars = ax.barh(methods, times, color=['#d9534f', '#5cb85c'], height=0.5, edgecolor='black')
ax.set_xscale('log') 
ax.set_xlabel('Computational Time (Seconds, Log Scale)')
ax.set_title('Inference Time Comparison\n(Finding 1 Target Geometry)')
ax.grid(True, axis='x', which='both', alpha=0.3)
ax.text(cst_time_seconds * 1.2, 0, f"~{cst_time_seconds/60:.0f} mins", va='center')
ax.text(ml_time_seconds * 1.2, 1, f"{ml_time_seconds*1000:.1f} ms", va='center')
plt.tight_layout()
plt.savefig('computational_speedup.png', dpi=180, bbox_inches='tight')
plt.close()


#BATCH PREDICTION

target_Qs = [1383]
target_freq = 2855.90

batch_input = np.array([[q, target_freq] for q in target_Qs])
predicted_lengths = length_predictor.predict(batch_input)
predicted_radii   = radius_predictor.predict(batch_input)

print("\n" + "="*75)
print(f" BATCH PREDICTIONS (Target Freq = {target_freq} MHz)")
print("="*75)
print(f"{'Target Q-factor':<18} | {'Predicted Length (mm)':<25} | {'Predicted Radius (mm)'}")
print("-" * 75)
for q, length, radius in zip(target_Qs, predicted_lengths, predicted_radii):
    print(f"{q:<18} | {length:<25.4f} | {radius:.4f}")
print("=" * 75 + "\n")





