import pandas as pd
import numpy as np
from scipy.optimize import minimize
from prophet import Prophet
import warnings
import datetime

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess datasets from CSV files.
    """
    # Load datasets
    products = pd.read_csv('data/products.csv')
    customers = pd.read_csv('data/customers.csv')
    transactions = pd.read_csv('data/transactions.csv')
    market_trends = pd.read_csv('data/market_trends.csv')
    competitor_data = pd.read_csv('data/competitor_data.csv')
    marketing_campaigns = pd.read_csv('data/marketing_campaigns.csv')
    
    # Convert date columns to datetime objects without enforcing a fixed format,
    # so that both ISO8601 and dd-mm-yyyy formats can be parsed.
    transactions['DateTime'] = pd.to_datetime(transactions['DateTime'], errors='coerce')
    market_trends['DateTime'] = pd.to_datetime(market_trends['DateTime'], errors='coerce')
    competitor_data['DateTime'] = pd.to_datetime(competitor_data['DateTime'], errors='coerce')
    products['Launch_Date'] = pd.to_datetime(products['Launch_Date'], errors='coerce')
    customers['Last_Purchase_Date'] = pd.to_datetime(customers['Last_Purchase_Date'], errors='coerce')
    marketing_campaigns['Start_Date'] = pd.to_datetime(marketing_campaigns['Start_Date'], errors='coerce')
    marketing_campaigns['End_Date'] = pd.to_datetime(marketing_campaigns['End_Date'], errors='coerce')
    
    # Merge datasets based on key fields
    merged_data = transactions.merge(products, on='Product_ID', suffixes=('', '_prod')) \
                              .merge(customers, on='Customer_ID', suffixes=('', '_cust')) \
                              .merge(competitor_data, on=['Product_ID', 'DateTime'], how='left') \
                              .merge(market_trends, on='DateTime', how='left')
    
    # For marketing campaigns, assign campaign details if transaction date falls within campaign period
    current_date = datetime.datetime.now()
    active_campaigns = marketing_campaigns[(marketing_campaigns['Start_Date'] <= current_date) & 
                                             (marketing_campaigns['End_Date'] >= current_date)]
    if not active_campaigns.empty:
        # Assuming campaigns target all products; if not, join on Product_ID as an additional key
        campaign_discount = active_campaigns['Conversion_Rate'].mean()  # using conversion rate as a proxy for discount impact
        merged_data['Campaign_Discount'] = campaign_discount
    else:
        merged_data['Campaign_Discount'] = 0

    # Handle missing values
    merged_data.fillna({
        'Competitor_Base_Price': merged_data['Base_Price'],
        'Competitor_Final_Price': merged_data['Final_Price'],
        'Competitor_Discount': 0,
        'Google_Trend_Score': merged_data['Google_Trend_Score'].mean(),
        'Market_Sentiment': 0,
        'Seasonal_Impact': merged_data['Seasonal_Impact_prod'].fillna(merged_data['Seasonal_Impact'].mode()[0])
    }, inplace=True)
    
    return merged_data

def forecast_demand(data):
    """
    Forecast future demand for each product using Prophet.
    """
    demand_forecast = {}
    for product_id in data['Product_ID'].unique():
        # Aggregate quantity purchased over dates
        product_data = data[data['Product_ID'] == product_id][['DateTime', 'Quantity_Purchased']].rename(
            columns={'DateTime': 'ds', 'Quantity_Purchased': 'y'}
        )
        product_data = product_data.groupby('ds').sum().reset_index()
        if len(product_data) < 2:
            # Not enough data; use a default forecast of 1 unit per day
            demand_forecast[product_id] = 1
            continue

        # Use Prophet model for forecasting with seasonalities
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(product_data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        # Average forecasted demand over the next 30 days
        demand_forecast[product_id] = forecast[['ds', 'yhat']].tail(30).mean()['yhat']
    return demand_forecast

def calculate_price_elasticity(data):
    """
    Calculate price elasticity for each product.
    """
    elasticity = {}
    for product_id in data['Product_ID'].unique():
        product_data = data[data['Product_ID'] == product_id][['Base_Price', 'Quantity_Purchased']].sort_values('Base_Price')
        if len(product_data) > 1:
            price_pct_change = product_data['Base_Price'].pct_change()
            quantity_pct_change = product_data['Quantity_Purchased'].pct_change()
            elasticity_value = (quantity_pct_change / price_pct_change).mean()
            elasticity[product_id] = elasticity_value if np.isfinite(elasticity_value) else 0
        else:
            elasticity[product_id] = 0  # Insufficient data, default elasticity is 0
    return elasticity

def competitor_price_analysis(data):
    """
    Analyze competitor pricing and availability.
    """
    competitor_prices = data.groupby('Product_ID').agg({
        'Competitor_Final_Price': 'mean',
        'Competitor_Stock_Availability': 'mean'
    }).to_dict()
    return competitor_prices.get('Competitor_Final_Price', {}), competitor_prices.get('Competitor_Stock_Availability', {})

def optimize_price(product_data, elasticity, competitor_price, demand_forecast, stock_level, campaign_discount):
    """
    Optimize product pricing to maximize profit, incorporating business rules such as competitive constraints,
    inventory limitations, and campaign effects.
    """
    cost_price = product_data['Cost_Price'].iloc[0]
    storage_cost = product_data['Storage_Cost'].iloc[0]
    shipping_cost = product_data['Shipping_Cost'].iloc[0]
    base_price = product_data['Base_Price'].iloc[0]
    
    def objective(price):
        total_cost = cost_price + storage_cost + shipping_cost
        # Adjust base demand using campaign impact
        adjusted_campaign = 1 + campaign_discount
        # Apply scarcity premium if forecasted demand exceeds stock level
        scarcity_modifier = 1 if demand_forecast <= stock_level else 1.1
        # Calculate demand using elasticity and adjustments
        demand = demand_forecast * adjusted_campaign * scarcity_modifier * (1 + elasticity * ((price[0] - base_price) / base_price))
        profit = (price[0] - total_cost) * demand
        return -profit  # Negative profit for maximization

    # Business constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - (cost_price + storage_cost + shipping_cost)},
        {'type': 'ineq', 'fun': lambda x: competitor_price * 1.2 - x[0]},
        {'type': 'ineq', 'fun': lambda x: stock_level - demand_forecast * (1 + elasticity * ((x[0] - base_price) / base_price))}
    ]
    
    initial_price = [base_price]
    bounds = [(cost_price * 1.1, competitor_price * 1.5)]
    
    result = minimize(objective, initial_price, constraints=constraints, bounds=bounds, method='SLSQP')
    return result.x[0] if result.success else base_price

def personalized_pricing(data, optimal_prices):
    """
    Calculate personalized price adjustments based on customer segments.
    """
    personalized_prices = {}
    for product_id in data['Product_ID'].unique():
        product_customers = data[data['Product_ID'] == product_id]
        base_optimal_price = optimal_prices.get(product_id, product_customers['Base_Price'].iloc[0])
        
        for customer_id in product_customers['Customer_ID'].unique():
            customer_data = product_customers[product_customers['Customer_ID'] == customer_id]
            discount_sensitivity = customer_data['Discount_Sensitivity'].iloc[0]
            loyalty_score = customer_data['Loyalty_Score'].iloc[0]
            
            if discount_sensitivity == 'High':
                personalized_prices[(product_id, customer_id)] = base_optimal_price * 0.9  # 10% discount
            elif loyalty_score > 80:
                personalized_prices[(product_id, customer_id)] = base_optimal_price * 0.95  # 5% discount for loyal customers
            else:
                personalized_prices[(product_id, customer_id)] = base_optimal_price
    return personalized_prices

def dynamic_pricing_engine():
    """
    Main function that runs the advanced dynamic pricing engine.
    """
    data = load_and_preprocess_data()
    demand_forecast = forecast_demand(data)
    elasticity = calculate_price_elasticity(data)
    competitor_prices, competitor_stock = competitor_price_analysis(data)
    
    optimal_prices = {}
    for product_id in data['Product_ID'].unique():
        product_data = data[data['Product_ID'] == product_id]
        comp_price = competitor_prices.get(product_id, product_data['Base_Price'].mean())
        stock_level = product_data['Stock_Level'].iloc[0]
        campaign_discount = product_data['Campaign_Discount'].iloc[0]
        optimal_prices[product_id] = optimize_price(
            product_data,
            elasticity.get(product_id, 0),
            comp_price,
            demand_forecast.get(product_id, 1),
            stock_level,
            campaign_discount
        )
    
    personalized_prices = personalized_pricing(data, optimal_prices)
    
    pricing_df = pd.DataFrame(list(optimal_prices.items()), columns=['Product_ID', 'Optimal_Price'])
    personalized_df = pd.DataFrame([(k[0], k[1], v) for k, v in personalized_prices.items()],
                                   columns=['Product_ID', 'Customer_ID', 'Personalized_Price'])
    
    pricing_df.to_csv('optimal_prices.csv', index=False)
    personalized_df.to_csv('personalized_prices.csv', index=False)
    
    print("Advanced dynamic pricing completed. Results saved to 'optimal_prices.csv' and 'personalized_prices.csv'.")

if __name__ == "__main__":
    dynamic_pricing_engine()