import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn as sk
teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
city=['Ahmedabad', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai', 'Sharjah',
       'Abu Dhabi', 'Delhi', 'Chennai','Hyderabad', 'Visakhapatnam',
       'Chandigarh', 'Bengaluru', 'Kolkata', 'Jaipur', 'Indore',
       'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur',
       'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
       'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']
pipe=pickle.load(open('pipe.pkl','rb'))
st.title('IPL WIN PROBABILITY PREDICTER')
col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox('Select Batting Team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select Bowling Team',sorted(teams))
selected_city=st.selectbox('Select Host City',sorted(city))
target=st.number_input('Target')
col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets')
if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    remaining_wickets=10-wickets
    crr=score/overs
    rrr=(runs_left*6)/balls_left
    input_df=pd.DataFrame({'BattingTeam':[batting_team],'Team1':[bowling_team],
                           'City':[selected_city],'runs_left':[runs_left],
                           'remaining_balls':[balls_left],'remaining_wickets':[remaining_wickets],
                           'target':[target],'crr':[crr],'rrr':[rrr]})
    st.table(input_df)
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team +':'+ str(round(win*100))+'%')
    st.header(bowling_team +':'+ str(round(loss*100))+'%')

