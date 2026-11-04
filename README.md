# Health-Buddy

A Streamlit app that signs the user in with Google OAuth and reads their Google Fit data (steps, height and weight) and basic profile details (birthday and gender) from the People API. It shows a BMI gauge and body composition chart and recommends workouts with YouTube links from a small CSV dataset based on gender, age, height and weight. A diet plan section is marked as coming soon.

## Tech stack

- Python, Streamlit
- Google OAuth 2.0 (oauthlib), Google Fit and People APIs
- pandas, Plotly

## How to run

1. Create OAuth credentials in Google Cloud with the Fitness and People APIs enabled.
2. Add them to `.streamlit/secrets.toml`:

   ```toml
   [general]
   CLIENT_ID = "..."
   CLIENT_SECRET = "..."
   REDIRECT_URI = "http://localhost:8501"
   ```

3. Install dependencies and start the app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Structure

- `app.py` - Streamlit app, OAuth flow and Google API calls
- `workouts.csv` - workout recommendations by gender, age, height and weight
- `requirements.txt` - dependencies
