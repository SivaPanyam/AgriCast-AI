import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from statsmodels.tsa.statespace.sarimax import SARIMAX

_CACHED_MODEL_BUNDLE = None

def get_model_bundle(base_dir: Path):
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE

    model_path = base_dir / 'models' / 'sarima_model.pkl'
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                _CACHED_MODEL_BUNDLE = pickle.load(f)
                return _CACHED_MODEL_BUNDLE
        except Exception as e:
            print(f"Warning: Failed to load pickled model: {e}")

    return None


@api_view(['GET', 'POST'])
def generate_forecast(request):
    try:
        base_dir = Path(settings.BASE_DIR)
        bundle = get_model_bundle(base_dir)

        if bundle is None:
            return Response(
                {'error': 'Model bundle not found. Please train models first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        steps = int(request.GET.get('steps', 12) if request.method == 'GET' else request.data.get('steps', 12))
        selected_model_type = request.GET.get('model', 'sarima').lower() if request.method == 'GET' else request.data.get('model', 'sarima').lower()

        if steps < 1 or steps > 36:
            steps = 12

        # Check for custom uploaded CSV records
        custom_series = None
        if request.method == 'POST' and 'custom_data' in request.data:
            # Expecting list of {date: str, price: float}
            records = request.data['custom_data']
            if len(records) >= 24:
                custom_df = pd.DataFrame(records)
                custom_df['date'] = pd.to_datetime(custom_df['date'], errors='coerce')
                custom_df['price'] = pd.to_numeric(custom_df['price'], errors='coerce')
                custom_df = custom_df.dropna().sort_values('date').drop_duplicates('date', keep='last')
                custom_series = custom_df.set_index('date')['price'].asfreq('MS')

        ts = custom_series if custom_series is not None else bundle['historical_series']

        # Determine model to use
        sarima_model = bundle.get('sarima_model') or bundle.get('fitted_model')
        hw_model = bundle.get('hw_model')
        xgb_artifact = bundle.get('xgb_artifact')

        if custom_series is not None:
            # Fit dynamic SARIMA for custom uploaded data
            sarima_model = SARIMAX(
                ts,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

        # Generate SARIMA Forecast
        forecast_result = sarima_model.get_forecast(steps=steps)
        forecast_df = forecast_result.summary_frame()

        sarima_forecasts = []
        for date, row in forecast_df.iterrows():
            sarima_forecasts.append({
                'date': date.strftime('%Y-%m-%d'),
                'forecast': round(float(row['mean']), 2),
                'lower_ci': round(float(row['mean_ci_lower']), 2),
                'upper_ci': round(float(row['mean_ci_upper']), 2)
            })

        # Generate Holt-Winters Forecast
        hw_forecasts = []
        if hw_model is not None and custom_series is None:
            try:
                hw_preds = hw_model.forecast(steps=steps)
                for date, val in hw_preds.items():
                    hw_forecasts.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'forecast': round(float(val), 2),
                        'lower_ci': round(float(val * 0.90), 2),
                        'upper_ci': round(float(val * 1.10), 2)
                    })
            except Exception:
                hw_forecasts = sarima_forecasts
        else:
            hw_forecasts = sarima_forecasts

        # Generate XGBoost Forecast (recursive multi-step)
        xgb_forecasts = []
        if xgb_artifact is not None and custom_series is None:
            try:
                xgb_reg = xgb_artifact['model']
                window = list(xgb_artifact['last_window'])
                curr_date = ts.index[-1]
                for _ in range(steps):
                    curr_date += pd.DateOffset(months=1)
                    # Lag features 1 to 12
                    feat = np.array(window[-12:][::-1]).reshape(1, -1)
                    pred = float(xgb_reg.predict(feat)[0])
                    window.append(pred)
                    xgb_forecasts.append({
                        'date': curr_date.strftime('%Y-%m-%d'),
                        'forecast': round(pred, 2),
                        'lower_ci': round(pred * 0.88, 2),
                        'upper_ci': round(pred * 1.12, 2)
                    })
            except Exception:
                xgb_forecasts = sarima_forecasts
        else:
            xgb_forecasts = sarima_forecasts

        # Pick primary forecast list based on model selected
        if selected_model_type == 'holtwinters':
            primary_forecasts = hw_forecasts
        elif selected_model_type == 'xgboost':
            primary_forecasts = xgb_forecasts
        else:
            primary_forecasts = sarima_forecasts

        # Historical series tail
        recent_history = []
        for date, val in ts.tail(36).items():
            recent_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(val), 2)
            })

        return Response({
            'success': True,
            'model': selected_model_type.upper(),
            'training_observations': int(len(ts)),
            'last_historical_date': ts.index[-1].strftime('%Y-%m-%d'),
            'last_price': float(ts.iloc[-1]),
            'forecast_horizon': steps,
            'metrics': bundle.get('metrics', {}),
            'history': recent_history,
            'forecasts': primary_forecasts,
            'comparison': {
                'sarima': sarima_forecasts,
                'holtwinters': hw_forecasts,
                'xgboost': xgb_forecasts
            },
            'seasonality_profile': bundle.get('seasonality_profile', []),
            'drift_monitoring': bundle.get('drift_monitoring', {})
        })

    except Exception as e:
        return Response(
            {
                'success': False,
                'error': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
