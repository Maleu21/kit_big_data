import streamlit as st
import plotly.express as px
import pandas as pd

# Example data
df = pd.DataFrame({
    'x': range(10),
    'y': range(10)
})

# Example plot
fig = px.line(df, x='x', y='y')

# Streamlit app
st.title('My Streamlit App')
st.plotly_chart(fig)