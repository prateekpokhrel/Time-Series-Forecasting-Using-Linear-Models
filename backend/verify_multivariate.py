"""
Quick end-to-end smoke-test for the multivariate (5-feature) pipeline.
Run from the backend directory:
    python verify_multivariate.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.forecasting import run_stacked_forecast

print("Running 30d multivariate forecast for RELIANCE.NS ...")
result = run_stacked_forecast("RELIANCE.NS", horizon="30d", data_source="yfinance")

print("\n=== Stacking Weights ===")
for k, v in result["stack_weights"].items():
    print(f"  {k:10s}: {v*100:.2f}%")

print("\n=== Model MSE ===")
for k, v in result["model_mse"].items():
    print(f"  {k:10s}: {v:.8f}")

print("\n=== Sample Forecast (first 5 days) ===")
for row in result["forecast"][:5]:
    print(f"  {row}")

print("\n✅ Multivariate pipeline OK!")
print(f"   Source: {result['source']}")
print(f"   N forecast points: {len(result['forecast'])}")
