import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import CapstoneRidgeModel
from sklearn.metrics import pairwise_distances_argmin_min

df = pd.read_csv('imputed_data.csv')
drop_country_pisa = ['Country', 'PISA Score']
X = df.drop(columns=drop_country_pisa)
y = df['PISA Score']
PISA_average = df['PISA Score'].mean()

# Streamlit title
st.title('Classroom Factors Correlated with International Test (PISA) Score')

# Dropdown menu for selecting a feature
selected_feature = st.selectbox('Choose a feature to explore correlation:', df.columns[1:])

correlation = np.corrcoef(df[selected_feature], df['PISA Score'])[0, 1]
r_squared = correlation ** 2

# Display the selected feature's data
st.write(f"{selected_feature}")

alt_chart = alt.Chart(df).mark_circle().encode(
    x=alt.X(selected_feature, scale=alt.Scale(zero=False)),  # Disable zero baseline on x-axis
    y=alt.Y('PISA Score', scale=alt.Scale(padding=0.1, zero=False)),  # Add padding to the y-axis
    color='Label:N'
)

# Add dynamic regression line based on selected feature and PISA Score
regression_line = alt_chart.transform_regression(
    selected_feature, 'PISA Score', method='linear'
).mark_line(color='red')

# Combine scatter plot and regression line
alt_chart = (alt_chart + regression_line).properties(width=600, height=400)
st.altair_chart(alt_chart, use_container_width=True)

# Ask user to conduct Teaching and Learning International Survey at their own school.
st.write(
    "If you want to see how your district compares, please administer the Teaching and Learning International Survey in your own school and input your results in the sidebar. Click 'Predict Performance' below when you're done.")

# User inputs results as percentage
predict_button = st.button('Predict Performance!')

# Sidebar
st.sidebar.title("How does your school compare?")

# Create an empty dictionary to store user input
# This is important to assign user input to feature names
user_input_dict = {}

# Collect user input for each feature
for feature_name in df.columns[1:81]:
    default = int(df[feature_name].mean())
    user_value = st.sidebar.number_input(f'Enter percent of {feature_name}.', min_value=0, max_value=100, value=default)
    user_input_dict[feature_name] = user_value

# Convert user input dictionary to a DataFrame for prediction
user_input_df = pd.DataFrame([user_input_dict])

# load trained model from ridge model script
loaded_model = CapstoneRidgeModel.create_train_model()

prediction_placeholder = st.empty()
# Display prediction if button is clicked
if predict_button:
    # make prediction
    predicted_value = loaded_model.predict(user_input_df)[0]

    prediction_placeholder.subheader('Predicted PISA Score:')
    prediction_placeholder.write(f'Based on your input, the predicted PISA Score is: {predicted_value:.2f}. The average score of 48 other well-developed countries in 2023 is {PISA_average:.2f}')

    # pairwise outputs an array, with the index, and it's distance. need to only pull the index
    closest_indices = pairwise_distances_argmin_min(user_input_df,X)[0]
    most_similar_country = df.loc[closest_indices[0]][0]
    most_similar_country_score = df.loc[closest_indices[0]]['PISA Score']

    # find the country with the most similar score
    closest_country_index = (df['PISA Score'] - predicted_value).abs().idxmin()
    similar_country = df.loc[closest_country_index][0]
    similar_country_score = df.loc[closest_country_index]['PISA Score']

    # Similar countries
    st.write(f"The country most similar to your results is {most_similar_country}, with a score of {most_similar_country_score:.2f}, and the country that scored similarly to you is {similar_country}, with a score of {similar_country_score:.2f}")
    st.write("Please hover over the bars below for more information")

    # dataframe for visualization
    data = pd.DataFrame({
        "Category": ['Predicted Score', f"Most similar country:\n{most_similar_country}", f"Similarly scoring country:\n{similar_country}"],
        "PISA Score": [predicted_value, most_similar_country_score, similar_country_score]
    })

    bar_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Category", axis=alt.Axis(labelAngle=0)),  # Keep the labels horizontal
        y=alt.Y("PISA Score"),
        text=alt.Text("PISA Score:N", format=".2f")
    )

    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader('How to improve:')
    # Provides recommendation on improvement of school
    st.write("Here are your school's biggest factors that determines your PISA Score.")
    st.write("The greater the value, the larger impact this factor has on your result.")

print(min(df['PISA Score']))
