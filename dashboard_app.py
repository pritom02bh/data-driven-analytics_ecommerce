import streamlit as st
import pandas as pd

@st.cache
def load_pricing_data():
    optimal = pd.read_csv('optimal_prices.csv')
    personalized = pd.read_csv('personalized_prices.csv')
    return optimal, personalized

def main():
    st.title("Dynamic Pricing Optimization Dashboard")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Overview", "Optimal Prices", "Personalized Pricing"])
    
    optimal, personalized = load_pricing_data()
    
    if page == "Overview":
        st.header("Overview")
        st.write("""
        This dashboard provides insights into the advanced dynamic pricing engine.
        Key metrics include optimal product pricing recommendations and personalized pricing 
        adjustments for individual customer segments.
        """)
        st.subheader("Key Metrics")
        st.metric(label="Total Products", value=optimal.shape[0])
        st.metric(label="Personalized Price Rules", value=personalized.shape[0])
    
    elif page == "Optimal Prices":
        st.header("Optimal Prices")
        st.write("Below is the table of optimal prices recommended for each product:")
        st.dataframe(optimal)
        st.download_button("Download CSV", optimal.to_csv(index=False), "optimal_prices.csv", "text/csv")
    
    elif page == "Personalized Pricing":
        st.header("Personalized Pricing")
        st.write("This table shows personalized price adjustments for individual customers:")
        st.dataframe(personalized)
        st.download_button("Download CSV", personalized.to_csv(index=False), "personalized_prices.csv", "text/csv")
    
if __name__ == '__main__':
    main()