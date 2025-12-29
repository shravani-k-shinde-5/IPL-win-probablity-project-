import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="IPL WINNING PREDICTOR",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS for pastel background and highlight effects
st.markdown(
    """
    <style>
    /* Soft pastel background for entire app */
    body {
        background-color: #fef6e4;
        font-family: 'Arial', sans-serif;
    }

    /* Header with subtle shadow and highlight effect */
    .header {
        background-color: #ffd6d6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #ff4b4b;
        font-size: 2.2em;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(255,75,75,0.3);
        transition: all 0.3s ease-in-out;
    }
    .header:hover {
        box-shadow: 0 0 25px rgba(255,75,75,0.6);
    }

    /* Button highlight effect */
    div.stButton > button {
        background-color: #ffd6d6;
        color: #ff4b4b;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #ffb3b3;
        transform: scale(1.05);
    }

    /* Input boxes highlight on focus */
    input, .stSelectbox, .stNumberInput {
        border-radius: 8px;
        border: 1px solid #ffb3b3;
        box-shadow: 0 0 8px rgba(255,75,75,0.2);
        transition: all 0.3s ease;
    }
    input:focus {
        box-shadow: 0 0 15px rgba(255,75,75,0.5);
        border-color: #ff4b4b;
    }

    /* Metrics with highlight glow */
    .stMetric {
        border-radius: 15px;
        padding: 15px;
        background-color: #ffe0e0;
        color: #ff4b4b;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(255,75,75,0.3);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        box-shadow: 0 0 25px rgba(255,75,75,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">🏏 IPL WINNING PREDICTOR 🏏</div>', unsafe_allow_html=True)
st.write("")

# Teams and city
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]



pipe = pickle.load(open("pipe.pkl", "rb"))


# Columns for selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox("Select host city", sorted(cities))

# Match stats
target = st.number_input('Target', min_value=0)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Score", min_value=0)
with col4:
    over = st.number_input("Over completed", min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input("Wickets out", min_value=0, max_value=10)

# Prediction button
if st.button("Predict probability of win"):
    # Calculations
    runs_left = target - score
    ball_left = 120 - (over * 6)
    wicket_left = 10 - wickets_out
    crr = score / over if over > 0 else 0
    rrr = (runs_left * 6) / ball_left if ball_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'ball_left': [ball_left],
        'wickets': [wicket_left],
        'total_runs_x': [target],
        'curr_rate': [crr],
        'rrr_rate': [rrr]
    })

    # Input table
    st.markdown('<h3 style="color:#FF4B4B;">Match Input Data:</h3>', unsafe_allow_html=True)
    st.table(input_df)

    # Prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Metrics with pastel highlight
    col_win, col_loss = st.columns(2)
    col_win.metric(label=f"{batting_team} Winning Probability", value=f"{round(win*100)}%")
    col_loss.metric(label=f"{bowling_team} Winning Probability", value=f"{round(loss*100)}%")
