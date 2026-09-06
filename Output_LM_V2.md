
====================================================================================================
  COMPREHENSIVE LTV PREDICTION ANALYSIS - START
  Timestamp: 2026-04-14 11:35:45
====================================================================================================

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 1: DATA LOADING & PREPARATION
════════════════════════════════════════════════════════════════════════════════════════════════════

  Initial Dataset:
    • Shape: (50000, 25)
    • Columns: 25
    • Missing values: 49081

  Missing value handling:
    • Age: 2495 values → median imputation
    • Session_Duration_Avg: 3399 values → median imputation
    • Pages_Per_Session: 3000 values → median imputation
    • Wishlist_Items: 4000 values → median imputation
    • Days_Since_Last_Purchase: 3000 values → median imputation
    • Discount_Usage_Rate: 3500 values → median imputation
    • Returns_Rate: 4491 values → median imputation
    • Email_Open_Rate: 2528 values → median imputation
    • Customer_Service_Calls: 168 values → median imputation
    • Product_Reviews_Written: 3500 values → median imputation
    • Social_Media_Engagement_Score: 6000 values → median imputation
    • Mobile_App_Usage: 5000 values → median imputation
    • Payment_Method_Diversity: 2500 values → median imputation
    • Credit_Balance: 5500 values → median imputation
  ✓ Remaining NaN values: 0

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 2: FEATURE ENGINEERING
════════════════════════════════════════════════════════════════════════════════════════════════════

  Creating new features:
    ✓ Engagement_Index = Session_Duration_Avg × Pages_Per_Session
    ✓ Purchase_Recency = 1 / (Days_Since_Last_Purchase + 1)
    ✓ Loyalty_Ratio = Total_Purchases / (Membership_Years + 1)

  New feature statistics:
       Engagement_Index  Purchase_Recency  Loyalty_Ratio
count      50000.000000      50000.000000   50000.000000
mean         268.088199          0.097627       4.293342
std          200.846012          0.154071       4.126543
min            1.100000          0.003472      -7.222222
25%          123.497500          0.025000       2.000000
50%          217.500000          0.045455       3.333333
75%          357.485000          0.100000       5.454545
max         1572.620000          1.000000     112.703358

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 3: DESCRIPTIVE STATISTICS & EXPLORATORY ANALYSIS
════════════════════════════════════════════════════════════════════════════════════════════════════

  Target Variable (Lifetime_Value) Statistics:
    • Mean: $1440.63
    • Median: $1243.41
    • Std Dev: $907.25
    • Min: $0.00
    • Max: $8987.24
    • Skewness: 1.4474
    • Kurtosis: 3.2736

  Numeric Features Summary:
    • Total numeric features: 24
    • Features with high skewness (|skew| > 1): 11

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 4: MODEL DATA PREPARATION
════════════════════════════════════════════════════════════════════════════════════════════════════

  Final data cleaning:
  ✓ No NaN values in final dataset

  Data shapes:
    • Features (X): (50000, 27)
    • Target (y): (50000,)
    • Number of predictors: 27

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 4A: MULTICOLLINEARITY CHECK (VARIANCE INFLATION FACTOR)
════════════════════════════════════════════════════════════════════════════════════════════════════

  VIF Analysis (Multicollinearity Assessment):
  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
  VIF Interpretation:
    • VIF < 5: Low multicollinearity ✓
    • VIF 5-10: Moderate multicollinearity ⚠
    • VIF > 10: High multicollinearity ✗

  Top 15 Features by VIF Score:
                      Feature       VIF
             Engagement_Index 20.088926
            Pages_Per_Session  8.357332
         Session_Duration_Avg  7.726824
              Total_Purchases  6.115983
                Loyalty_Ratio  4.823985
             Mobile_App_Usage  2.566181
              Login_Frequency  2.452335
        Cart_Abandonment_Rate  2.377028
              Email_Open_Rate  2.058100
               Wishlist_Items  1.980769
             Membership_Years  1.953376
Social_Media_Engagement_Score  1.824709
      Product_Reviews_Written  1.690890
            Signup_Quarter_Q4  1.686338
               Credit_Balance  1.510860

  Summary:
    • Features with VIF > 10: 1
    • Features with VIF 5-10: 3
    • Features with VIF < 5: 23
    ⚠ Warning: 1 features show high multicollinearity
      Ridge regression may help regularize these relationships

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 5: OLS REGRESSION (ORIGINAL VARIABLES)
════════════════════════════════════════════════════════════════════════════════════════════════════

  Model Performance:
    • R² Score: 0.505196
    • Adjusted R²: 0.504929
    • RMSE: $638.17
    • MAE: $444.63
    • MAPE: 149685734704167.375000

  Residual Analysis:
    • Mean: -0.000000
    • Std Dev: 638.180016
    • Skewness: 0.773527
    • Kurtosis: 8.853300

  10-Fold Cross-Validation:
    • Mean R²: 0.502815
    • Std Dev: 0.016638
    • Fold scores: ['0.4992', '0.5199', '0.4859', '0.5001', '0.5300', '0.5132', '0.4732', '0.4854', '0.5077', '0.5137']

────────────────────────────────────────────────────────────────────────────────────────────────────
  → Statistical Significance Testing (OLS - Original Variables)
────────────────────────────────────────────────────────────────────────────────────────────────────

  Overall Model F-Test:
    • F-Statistic: 1889.687999
    • P-Value: 1.11e-16
    • Result: ✓ Model is statistically significant (p < 0.05)

  Top 10 Significant Variables (by absolute T-statistic):
  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
               Variable  Coefficient  Std Error  T-Statistic      P-Value Significant
        Total_Purchases    97.444987   1.006102    96.853950 0.000000e+00         Yes
    Average_Order_Value     0.771685   0.016265    47.444459 0.000000e+00         Yes
          Loyalty_Ratio   -62.402539   1.519485   -41.068219 0.000000e+00         Yes
      Signup_Quarter_Q4  -275.269281   8.568771   -32.124710 0.000000e+00         Yes
       Membership_Years   -51.449983   1.937735   -26.551615 0.000000e+00         Yes
           Returns_Rate   -14.761900   0.558920   -26.411451 0.000000e+00         Yes
                  const   423.401926  34.496093    12.273910 0.000000e+00         Yes
  Cart_Abandonment_Rate    -1.778799   0.270315    -6.580471 4.735812e-11         Yes
Product_Reviews_Written     9.415574   1.645150     5.723230 1.051138e-08         Yes
         Wishlist_Items     7.303575   1.312786     5.563415 2.658979e-08         Yes

  Summary: 16 out of 27 variables are significant (p < 0.05)

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 6: TRANSFORMATIONS FOR HOMOSCEDASTICITY
════════════════════════════════════════════════════════════════════════════════════════════════════

  Original Target Skewness: 1.447360
  Log-transformed Target Skewness: -0.668242
  ✓ Log Transform applied (target): Skewness improvement confirmed

  Identifying right-skewed features (|skew| > 1)...
    Found 14 right-skewed features
    ✓ Applied Yeo-Johnson to 14 features

  Applying Winsorization (1st-99th percentile)...
    ✓ Clipped outliers for all numeric features

  Standardizing features (mean=0, std=1)...
    ✓ Scaled all 27 features

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 7: OLS REGRESSION (TRANSFORMED VARIABLES)
════════════════════════════════════════════════════════════════════════════════════════════════════

  Model Performance:
    • R² Score: 0.913805
    • Adjusted R²: 0.913758
    • RMSE: 0.194780
    • MAE: 0.130081
    • MAPE: 4476748991469.983398

  Residual Analysis:
    • Mean: -0.000000
    • Std Dev: 0.194782
    • Skewness: -5.455433
    • Kurtosis: 136.422750

  10-Fold Cross-Validation:
    • Mean R²: 0.913734
    • Std Dev: 0.008596

────────────────────────────────────────────────────────────────────────────────────────────────────
  → Statistical Significance Testing (OLS - Transformed Variables)
────────────────────────────────────────────────────────────────────────────────────────────────────

  Overall Model F-Test:
    • F-Statistic: 19621.607786
    • P-Value: 1.11e-16
    • Result: ✓ Model is statistically significant (p < 0.05)

  Top 10 Significant Variables (by absolute T-statistic):
  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
            Variable  Coefficient  Std Error  T-Statistic  P-Value Significant
               const     7.075994   0.000871  8120.927040      0.0         Yes
 Average_Order_Value     0.371186   0.000872   425.903410      0.0         Yes
   Signup_Quarter_Q4    -0.118667   0.001138  -104.262565      0.0         Yes
     Total_Purchases     0.327388   0.004069    80.465758      0.0         Yes
       Loyalty_Ratio     0.275856   0.005367    51.400487      0.0         Yes
    Membership_Years     0.193264   0.003802    50.832749      0.0         Yes
    Engagement_Index     0.300083   0.006220    48.246597      0.0         Yes
Session_Duration_Avg    -0.161316   0.003512   -45.937307      0.0         Yes
   Pages_Per_Session    -0.174053   0.003898   -44.650669      0.0         Yes
      Wishlist_Items     0.013199   0.001228    10.745697      0.0         Yes

  Summary: 15 out of 27 variables are significant (p < 0.05)

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 8: RIDGE REGRESSION (TRANSFORMED VARIABLES)
════════════════════════════════════════════════════════════════════════════════════════════════════

  Tuning alpha parameter (10-fold CV)...
    ✓ Optimal alpha: 1.757511

  Model Performance:
    • R² Score: 0.913805
    • Adjusted R²: 0.913758
    • RMSE: 0.194780
    • MAE: 0.130078
    • MAPE: 4476997837935.127930

  10-Fold Cross-Validation:
    • Mean R²: 0.913734
    • Std Dev: 0.008596

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 9: PRINCIPAL COMPONENT REGRESSION
════════════════════════════════════════════════════════════════════════════════════════════════════

  PCA Analysis:
    • Total features: 27
    • Variance threshold: 90%
    • Selected components: 17
    • Variance explained: 0.9036
    • Reduced feature space: (50000, 17)

  Model Performance:
    • R² Score: 0.763672
    • Adjusted R²: 0.763591
    • RMSE: 0.322524
    • MAE: 0.236076
    • MAPE: 4850967879369.543945

  10-Fold Cross-Validation:
    • Mean R²: 0.763364
    • Std Dev: 0.009274

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 10: RANDOM FOREST REGRESSION
════════════════════════════════════════════════════════════════════════════════════════════════════

  Model Performance:
    • R² Score: 0.976600
    • Adjusted R²: 0.976587
    • RMSE: 0.101488
    • MAE: 0.069347
    • MAPE: 3157718376614.373047

  10-Fold Cross-Validation:
    • Mean R²: 0.948990
    • Std Dev: 0.007659

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 12: COMPREHENSIVE METRICS COMPARISON TABLE
════════════════════════════════════════════════════════════════════════════════════════════════════

            Model R² Score  Adj. R²       RMSE        MAE                   MAPE CV R² Mean CV R² Std Residual Skew Residual Kurt
   OLS (Original) 0.505196 0.504929 638.173635 444.630042 149685734704167.375000   0.502815  0.016638      0.773527      8.853300
OLS (Transformed) 0.913805 0.913758   0.194780   0.130081   4476748991469.983398   0.913734  0.008596     -5.455433    136.422750
            Ridge 0.913805 0.913758   0.194780   0.130078   4476997837935.127930   0.913734  0.008596     -5.456708    136.444034
              PCR 0.763672 0.763591   0.322524   0.236076   4850967879369.543945   0.763364  0.009274     -1.631957     24.041313
    Random Forest 0.976600 0.976587   0.101488   0.069347   3157718376614.373047   0.948990  0.007659    -10.734781    438.092214

  ✓ Saved to: model_metrics_comparison.csv

════════════════════════════════════════════════════════════════════════════════════════════════════
  STEP 13: GENERATING VISUALIZATIONS
════════════════════════════════════════════════════════════════════════════════════════════════════
  ✓ 00_vif_multicollinearity.png
  ✓ 01_ols_original_coefficients.png
  ✓ 01b_ols_transformed_coefficients.png
  ✓ 02_r2_comparison.png
  ✓ 03_error_metrics_comparison.png
  ✓ 04_crossvalidation_results.png
  ✓ 05_predicted_vs_actual.png
  ✓ 06_residual_distributions.png
  ✓ 07_qq_plots.png
  ✓ 08_feature_importance.png
  ✓ 09_adj_r2_comparison.png
  ✓ 10_residuals_comparison.png

  Total visualizations created: 12

════════════════════════════════════════════════════════════════════════════════════════════════════
  ANALYSIS COMPLETE
════════════════════════════════════════════════════════════════════════════════════════════════════

  ✓ All models trained and evaluated
  ✓ Multicollinearity analysis (VIF) completed
  ✓ Statistical significance tests (F-test, T-tests) performed for OLS models
  ✓ 10-fold cross-validation completed for all models
  ✓ Comprehensive metrics table created
  ✓ 11 professional visualizations generated

  Output files saved to: /mnt/user-data/outputs/
    • model_metrics_comparison.csv
    • 00_vif_multicollinearity.png
    • 01_ols_original_coefficients.png
    • 01b_ols_transformed_coefficients.png
    • 02_r2_comparison.png through 09_adj_r2_comparison.png

  Best performing model: Random Forest
    R² Score: 0.976600
