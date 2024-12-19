import streamlit as st
from google.oauth2 import id_token
from google.auth.transport import requests
from oauthlib.oauth2 import WebApplicationClient
import requests as req
from datetime import datetime, timedelta, date
import time
import pandas as pd
import plotly.graph_objects as go

SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/user.birthday.read",
    "https://www.googleapis.com/auth/user.gender.read",
    "https://www.googleapis.com/auth/fitness.body.write"
]

CLIENT_ID = st.secrets["general"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["general"]["CLIENT_SECRET"]
REDIRECT_URI = st.secrets["general"]["REDIRECT_URI"]
DISCOVERY_DOC = "https://accounts.google.com/.well-known/openid-configuration"

st.title("Health Buddy App")

# Load the dataset
# Make sure you have a workouts.csv file with columns:
# Gender,Age,Height (cm),Weight (kg),Workout,YouTube Link
df = pd.read_csv("workouts.csv")

# Initialize session state variables
if 'access_token' not in st.session_state:
    st.session_state['access_token'] = None
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

client = WebApplicationClient(CLIENT_ID)

try:
    discovery = req.get(DISCOVERY_DOC).json()
    auth_endpoint = discovery["authorization_endpoint"]
    login_url = client.prepare_request_uri(
        auth_endpoint,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES
    )
    if not st.session_state['logged_in']:
        st.markdown(f"[Login with Google]({login_url})")
except Exception as e:
    st.error(f"Failed to generate login URL: {e}")

code = st.query_params.get("code")

# Functions
def fetch_userinfo(headers, userinfo_endpoint):
    userinfo_response = req.get(userinfo_endpoint, headers=headers)
    if userinfo_response.status_code == 200:
        userinfo = userinfo_response.json()
        return userinfo.get("name", "N/A")
    return "N/A"

def fetch_people_data(headers):
    people_endpoint = "https://people.googleapis.com/v1/people/me"
    params = {"personFields": "birthdays,genders"}
    r = req.get(people_endpoint, headers=headers, params=params)
    if r.status_code == 200:
        person_data = r.json()
        genders = person_data.get("genders", [])
        user_gender = genders[0]["value"] if genders else "N/A"
        birthdays = person_data.get("birthdays", [])
        if birthdays:
            b = birthdays[0].get("date", {})
            year = b.get("year")
            month = b.get("month")
            day = b.get("day")
            if year and month and day:
                birthdate = date(year, month, day)
                today = date.today()
                user_age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            else:
                user_age = "N/A"
        else:
            user_age = "N/A"
        return user_age, user_gender
    return "N/A", "N/A"

def fetch_latest_data_point(headers, data_stream_id):
    start_time = datetime(2000, 1, 1)
    end_time = datetime.now()
    start_nanos = int(start_time.timestamp() * 1e9)
    end_nanos = int(end_time.timestamp() * 1e9)
    dataset_id = f"{start_nanos}-{end_nanos}"
    url = f"https://www.googleapis.com/fitness/v1/users/me/dataSources/{data_stream_id}/datasets/{dataset_id}"
    r = req.get(url, headers=headers)
    if r.status_code == 200:
        dataset = r.json()
        points = dataset.get("point", [])
        if points:
            points.sort(key=lambda p: int(p["endTimeNanos"]))
            latest_point = points[-1]
            values = latest_point.get("value", [])
            if values and "fpVal" in values[0]:
                return values[0]["fpVal"]
    return None

def write_data_point(headers, data_stream_id, value, data_type_name):
    now_ns = int(time.time() * 1e9)
    end_ns = now_ns + 1  # ensure non-zero duration
    url = f"https://www.googleapis.com/fitness/v1/users/me/dataSources/{data_stream_id}/datasets/{now_ns}-{end_ns}"

    body = {
      "dataSourceId": data_stream_id,
      "maxEndTimeNs": end_ns,
      "minStartTimeNs": now_ns,
      "point": [
        {
          "dataTypeName": data_type_name,
          "startTimeNanos": str(now_ns),
          "endTimeNanos": str(end_ns),
          "value": [{"fpVal": value}]
        }
      ]
    }

    r = req.patch(url, headers=headers, json=body)
    if r.status_code not in (200, 204):
        # Print detailed response for debugging
        print("Error writing data:", r.status_code, r.text)
    return r.status_code in (200, 204)

def get_existing_data_source_id(headers, data_type_name, data_stream_name):
    url = "https://www.googleapis.com/fitness/v1/users/me/dataSources"
    r = req.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        for ds in data.get("dataSource", []):
            if ds.get("dataType", {}).get("name") == data_type_name and ds.get("dataStreamName") == data_stream_name:
                return ds["dataStreamId"]
    return None

def create_data_source(headers, data_type_name, data_stream_name):
    url = "https://www.googleapis.com/fitness/v1/users/me/dataSources"
    body = {
        "dataStreamName": data_stream_name,
        "type": "raw",
        "application": {
            "name": "My Custom App"
        },
        "dataType": {
            "name": data_type_name
        }
    }

    existing_id = get_existing_data_source_id(headers, data_type_name, data_stream_name)
    if existing_id:
        return existing_id

    response = req.post(url, headers=headers, json=body)
    if response.status_code == 200:
        ds_data = response.json()
        return ds_data["dataStreamId"]
    else:
        print("Error creating data source:", response.status_code, response.text)
        return None

def fetch_steps_data(headers, days):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    dataset_endpoint = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    body = {
        "aggregateBy": [{"dataTypeName": "com.google.step_count.delta"}],
        "bucketByTime": {"durationMillis": 86400000},
        "startTimeMillis": start_time,
        "endTimeMillis": end_time,
    }
    dataset_response = req.post(dataset_endpoint, headers=headers, json=body)

    if dataset_response.status_code == 200:
        steps_data = dataset_response.json()
        total_steps = 0
        day_count = 0
        for bucket in steps_data.get("bucket", []):
            steps = sum(
                int(dp["value"][0]["intVal"]) for dp in bucket["dataset"][0].get("point", [])
            )
            total_steps += steps
            day_count += 1
        avg_steps = total_steps / day_count if day_count > 0 else 0
        return total_steps, avg_steps
    else:
        return None, None
        
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def calculate_composition(weight):
    # Sample estimation logic; replace with accurate methods if available
    bone_weight = 0.15 * weight
    muscle_weight = 0.35 * weight
    fat_weight = 0.25 * weight
    return bone_weight, muscle_weight, fat_weight

# Attempt token exchange only if we don't have a token yet and code is present
if code and st.session_state['access_token'] is None:
    try:
        with st.spinner("Loading user data..."):
            token_endpoint = discovery["token_endpoint"]
            userinfo_endpoint = discovery["userinfo_endpoint"]
            
            token_url, headers_, body = client.prepare_token_request(
                token_endpoint,
                redirect_url=REDIRECT_URI,
                code=code,
            )

            token_response = req.post(
                token_url,
                headers=headers_,
                data=body,
                auth=(CLIENT_ID, CLIENT_SECRET),
            )

            if token_response.status_code == 200:
                client.parse_request_body_response(token_response.text)
                st.session_state['access_token'] = token_response.json()["access_token"]
                st.session_state['logged_in'] = True
            else:
                st.error("Failed to exchange authorization code for access token.")
                st.stop()
    except Exception as e:
        st.error(f"Failed during token exchange: {e}")
        st.stop()

if st.session_state['access_token']:
    headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}
    try:
        with st.spinner("Fetching Google Fit data..."):
            userinfo_endpoint = discovery["userinfo_endpoint"]
            user_name = fetch_userinfo(headers, userinfo_endpoint)
            user_age, user_gender = fetch_people_data(headers)

            # Create or get existing data source IDs
            height_source_name = "My App com.google.height"
            weight_source_name = "My App com.google.weight"

            height_source_id = create_data_source(headers, "com.google.height", height_source_name)
            weight_source_id = create_data_source(headers, "com.google.weight", weight_source_name)

            user_height = "N/A"
            user_weight = "N/A"

            # Fetch height if we wrote it before
            if height_source_id:
                height_m = fetch_latest_data_point(headers, height_source_id)
                if height_m is not None:
                    user_height = round(height_m * 100, 2)  # convert m to cm

            # Fetch weight if we wrote it before
            if weight_source_id:
                weight_val = fetch_latest_data_point(headers, weight_source_id)
                if weight_val is not None:
                    user_weight = round(weight_val, 2)  # kg

            # Sidebar for navigation
            page_options = ["Home", "Workout Plan", "Diet Plan", "Body Composition", "Other Suggestions"]
            selected_page = st.sidebar.radio("Navigate", page_options)

            if selected_page == "Home":
                # Display User Details and forms
                st.write("### User Details")
                st.write(f"**Name**: {user_name}")
                st.write(f"**Age**: {user_age}")
                st.write(f"**Gender**: {user_gender}")
                st.write(f"**Height**: {user_height} {'cm' if user_height != 'N/A' else ''}")
                st.write(f"**Weight**: {user_weight} {'kg' if user_weight != 'N/A' else ''}")

                # Show forms only if values are N/A
                if user_height == "N/A" and height_source_id:
                    with st.form("height_form"):
                        new_height = st.number_input("Enter your height (cm)", min_value=50.0, max_value=300.0, value=50.0)
                        submit_height = st.form_submit_button("Save Height to Google Fit")
                        if submit_height:
                            height_m = new_height / 100.0
                            if write_data_point(headers, height_source_id, height_m, "com.google.height"):
                                st.success("Height updated in Google Fit! Please manually refresh to see changes.")
                            else:
                                st.error("Failed to update height.")

                if user_weight == "N/A" and weight_source_id:
                    with st.form("weight_form"):
                        new_weight = st.number_input("Enter your weight (kg)", min_value=10.0, max_value=300.0, value=10.0)
                        submit_weight = st.form_submit_button("Save Weight to Google Fit")
                        if submit_weight:
                            if write_data_point(headers, weight_source_id, new_weight, "com.google.weight"):
                                st.success("Weight updated in Google Fit! Please manually refresh to see changes.")
                            else:
                                st.error("Failed to update weight.")

                if user_gender == "N/A":
                    st.info("Gender cannot be updated via API. Please update it in your Google Account settings.")

                st.subheader("Steps Data for Multiple Durations:")
                durations = [365, 180, 90, 30, 7]
                steps_results = []
                for d in durations:
                    total_steps, avg_steps = fetch_steps_data(headers, d)
                    if total_steps is not None:
                        steps_results.append({
                            "Period (Days)": d,
                            "Total Steps": total_steps,
                            "Avg Steps/Day": round(avg_steps, 2)
                        })
                    else:
                        steps_results.append({
                            "Period (Days)": d,
                            "Total Steps": None,
                            "Avg Steps/Day": None
                        })

                df_steps_summary = pd.DataFrame(steps_results)
                st.dataframe(df_steps_summary)

            elif selected_page == "Workout Plan":
                st.header("Workout Plan")
                # Check if height or weight is missing
                if user_height == "N/A" or user_weight == "N/A" or user_age == "N/A" or user_gender == "N/A":
                    st.warning("Kindly update your height, weight, and ensure gender and age are available to recommend a workout.")
                else:
                    # Implement recommendation logic
                    # Filter the dataset by gender first
                    temp_df = df[df['Gender'].str.lower() == user_gender.lower()].copy()

                    # Function to try finding matches given an age
                    def find_workouts_for_age(search_age):
                        # Filter by this age
                        age_filtered = temp_df[temp_df['Age'] == search_age]
                        if age_filtered.empty:
                            return pd.DataFrame()

                        # Calculate difference metrics
                        # We'll find the closest matches by height and weight difference
                        age_filtered['height_diff'] = (age_filtered['Height (cm)'] - user_height).abs()
                        age_filtered['weight_diff'] = (age_filtered['Weight (kg)'] - user_weight).abs()
                        age_filtered['total_diff'] = age_filtered['height_diff'] + age_filtered['weight_diff']

                        # Sort by total_diff and take top 5
                        return age_filtered.sort_values('total_diff').head(5)

                    # Try exact age, then age-1, up to age-5
                    target_age = user_age
                    recommended = find_workouts_for_age(target_age)
                    if recommended.empty:
                        found = False
                        for i in range(1, 6):
                            new_age = target_age - i
                            recommended = find_workouts_for_age(new_age)
                            if not recommended.empty:
                                found = True
                                break
                        if not found:
                            # No suitable recommendations found
                            st.write("No suitable recommendations found.")
                        else:
                            st.write("### Recommended Workouts")
                            for _, row in recommended.iterrows():
                                st.write(f"**Workout**: {row['Workout']}")
                                st.video(row['YouTube Link'])
                                st.write("---")
                    else:
                        # We have recommendations for the exact age
                        st.write("### Recommended Workouts")
                        for _, row in recommended.iterrows():
                            st.write(f"**Workout**: {row['Workout']}")
                            st.video(row['YouTube Link'])
                            st.write("---")

            elif selected_page == "Diet Plan":
                st.header("Diet Plan")
                st.write("Diet plan suggestions coming soon...")

            elif selected_page == "Body Composition":
                st.header("Body Composition")

                if user_height == "N/A" or user_weight == "N/A":
                    st.warning("Kindly update your height and weight to view body composition.")
                else:
                    # Calculate BMI
                    bmi = calculate_bmi(user_weight, user_height)
            
                    # Calculate other compositions
                    bone_weight, muscle_weight, fat_weight = calculate_composition(user_weight)
            
                    # Display BMI as a gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=bmi,
                        title={'text': "BMI"},
                        gauge={
                            'axis': {'range': [0, 40], 'tickwidth': 1},
                            'bar': {'color': "red" if bmi > 30 else "green"},
                            'steps': [
                                {'range': [0, 18.5], 'color': "lightblue"},
                                {'range': [18.5, 25], 'color': "green"},
                                {'range': [25, 30], 'color': "yellow"},
                                {'range': [30, 40], 'color': "red"},
                            ],
                        }
                    ))
            
                    st.plotly_chart(fig, use_container_width=True)
            
                    # Display other compositions as a bar chart
                    composition_df = pd.DataFrame({
                        "Composition": ["Bone Weight", "Muscle Weight", "Fat Weight"],
                        "Weight (kg)": [bone_weight, muscle_weight, fat_weight]
                    })
            
                    st.bar_chart(composition_df.set_index("Composition"))
            
                    # Text summary
                    st.write(f"**Bone Weight**: {bone_weight:.2f} kg")
                    st.write(f"**Muscle Weight**: {muscle_weight:.2f} kg")
                    st.write(f"**Fat Weight**: {fat_weight:.2f} kg")

            elif selected_page == "Other Suggestions":
                st.header("Other Suggestions")
                st.write("Additional health suggestions coming soon...")

        st.success("Data loaded successfully!")

    except Exception as e:
        st.error(f"Failed during data fetch: {e}")
