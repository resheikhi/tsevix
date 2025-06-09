# VIX Calculator Dashboard

This Streamlit application calculates and displays VIX-like volatility index using options data. It provides real-time updates and historical visualization of the VIX values.

## Features
- Real-time VIX calculation
- Interactive update button
- Historical VIX chart
- Daily volatility estimates
- Raw options data display

## Installation

1. Clone this repository:
```bash
git clone https://github.com/resheikhi/tsevix.git
cd tsevix
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run vix5.py
```

## Usage
- Click the "Update VIX" button to fetch new data and update the VIX calculation
- View the historical VIX chart that updates with each new calculation
- Expand the "Show Option DataFrame" section to view the raw options data

## Note
Make sure you have a valid API token configured in the application for fetching options data. 