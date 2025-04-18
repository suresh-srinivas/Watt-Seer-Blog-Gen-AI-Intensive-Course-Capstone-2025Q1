
---
title: Watt-Saver Project Template
layout: default
---

# 🧠 Watt-Saver the Personalized Energy Coach
![header](/Image/header.png)

## This is the blog for [Gen AI Intensive Course Capstone 2025Q1 Porject](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1) which is part of [5-Day Gen AI Intensive Course with Google](https://rsvp.withgoogle.com/events/google-generative-ai-intensive_2025q1)


## 💡 Understanding Energy Bills with GenAI:

Many people receive energy bills without knowing why they’re high or how to reduce them. This project shows how Generative AI (GenAI) can help users understand their energy consumption and get personalized, human-readable feedback.

## ❓ Why/How GenAI can solve the problem?

Charts and numbers alone don’t help most people take action. GenAI transforms raw energy data into insightful narratives—like identifying your most expensive usage days and suggesting practical changes in behavior or device usage.

## 🔍 How It Works
We used hourly energy consumption data from a single home. After cleaning and summarizing it, we prompted Gemini (a GenAI model) to explain trends—like why usage spiked on certain days—and offer tips to reduce costs.

## 📊 Key Results
Instead of just visualizing data, Gemini responded with plain-language insights and personalized advice. It helped reveal things like:
- Which time of day is most expensive
- When heating/cooling is overused
- How to shift usage to off-peak hours

## 🚀 Impact
This approach can scale to households or communities. Tools like Watt-Save use GenAI to suggest smarter energy use—helping people cut costs and reduce waste without needing technical skills.
## 👥 Team Members

- [Jim](https://www.kaggle.com/jimkwikx)
- [Lamide](https://www.kaggle.com/olamidek)
- [Saad Asghar](https://www.kaggle.com/saadasghar) 
- [Jgar1597](https://www.kaggle.com/jgar1597)

## 💻 Notebook Link

[🔗 View on Kaggle](https://www.kaggle.com/code/jimkwikx/watt-saver-personalized-energy-jim?scriptVersionId=234347328)  

## 🛠️Code snippets to demonstrate how it was implemented

We analyzed the enegery consumption regarding the varity of factors on the dataset:  `Appliance Type`, `Energy Consumption (kWh)`, `Time`, `Date` ,`Outdoor Temperature (°C)`, `Season`  and `Household Size`.

We start of by loading the two datasets

```js
/kaggle/input/energy-data-meter-2/energy_hourly_data_2024_meter2.dat
/kaggle/input/smart-home-energy-consumption/smart_home_energy_consumption_large.csv
```

We will use `pd.read_csv` to display both datasets and get a overview of them:

![image2datasets](/Image/iamge-twodataasets.png)

Is it abit difficult to read the data therfore, let use GenAI to interpret the data for us, we will leverage the `Document understanding` capcability from [Day2](https://www.kaggle.com/code/markishere/day-2-document-q-a-with-rag)'s colab and `Image understaind` from [Bonus Day](https://www.kaggle.com/code/markishere/bonus-day-extra-api-features-to-try/notebook).


## Apply `Document understanding` to describe the datasets

Use `client.models.generate_content` to read the datasets and we set `teamperature 0.0` to make the output consistant.

```js
request = 'Tell me about the dataset document_file1 and document_file2,what are these data and the relationship bewteen them, give me code example'

def summarise_doc(request: str) -> str:
  """Execute the request on the uploaded document."""
  # Set the temperature low to stabilise the output.
  config = types.GenerateContentConfig(temperature=0.0)
  response = client.models.generate_content(
      model='gemini-2.0-flash',
      config=config,
      contents=[request, document_file1,document_file2], # Analyze the 2 dataset that defined above
  )

  return response.text
```
look at the following output, we can the the Gemini gave the overview of the datasets and some code snippets for analyzing them.

![image3](/Image/iamge3.png)
✅ The model gave a very quick overview of the datasets and how should we analysis them by providing code snippets: 

![image4](/Image/iamge4.png)
🤖 What code does is:
- Load Data: Reads both CSV and DAT files using pandas. Includes basic error handling for file loading.
- Preprocessing Dataset 1 and Dataset 2
- Estimate Cost per kWh: Calculates the cost per kWh from Dataset 2 (provided_cost / total_consumption). It uses the median for robustness and shows statistics/histogram to check if the rate is consistent (fixed).
- Compare Hourly vs. Appliance Data:
- Plots the comparison for a sample period.
- Identify High-Usage Appliances & Patterns:

✅ We can see the plot to have an quick grasp of the data and see Air conditioning and Heater have the highest energy consumption during the time period.
![prompt-analysis](/Image/prompt-analysis.png)

✅ and outdoor tempture vs consumption

![outdoor-tempture](/Image/Consumption-vs-outdoor-temp.png)

The second dataset is more meter data of a single household for one year on hourly basis and the other one is appliances of 500 households over multiple time period

We will first analysis this `energy_hourly_data_2024_meter2.dat` dataset to get the fixed rate


The second dataset is more meter data of a single household for one year on hourly basis and the other one is appliances of 500 households over multiple time period

We will first analysis this `energy_hourly_data_2024_meter2.dat` dataset to get the fixed rate.



```js
import pandas as pd

# Load with tab separator and column names
df = pd.read_csv(
    "/kaggle/input/energy-data-meter-2/energy_hourly_data_2024_meter2.dat",
    sep="\t",
    names=["start_time", "end_time", "consumption", "provided_cost", 
           "start_minus_prev_end", "end_minus_prev_end"],
    header=None,
    on_bad_lines='skip'
)

# Define exact datetime format (includes time zone offset)
datetime_format = "%Y-%m-%d %H:%M:%S%z"

# Explicit parse
df['start_time'] = pd.to_datetime(df['start_time'], utc=True,format=datetime_format, errors='coerce')
df['end_time'] = pd.to_datetime(df['end_time'], utc=True, format=datetime_format, errors='coerce')

# Drop bad rows
df.dropna(subset=['start_time', 'end_time'], inplace=True)

# Set index and select key columns
df.set_index('start_time', inplace=True)
df = df[['consumption', 'provided_cost']]

```
| start_time                |   consumption |   provided_cost |
|---------------------------|---------------|-----------------|
| 2024-01-01 08:00:00+00:00 |         0.377 |       0.0625707 |
| 2024-01-01 09:00:00+00:00 |         0.355 |       0.0589194 |
| 2024-01-01 10:00:00+00:00 |         0.352 |       0.0584214 |
| 2024-01-01 11:00:00+00:00 |         1.268 |       0.21045   |
| 2024-01-01 12:00:00+00:00 |         0.365 |       0.0605791 |

```js
# Ensure correct types
df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
df['provided_cost'] = pd.to_numeric(df['provided_cost'], errors='coerce')
# Resample hourly data to daily totals (IMPORTANT)
daily_data = df.resample('D').agg({
    'consumption': 'sum',
    'provided_cost': 'sum'
})
print(daily_data.head())
```

| start_time                |   consumption |   provided_cost |
|---------------------------|---------------|-----------------|
| 2024-01-01 00:00:00+00:00 |        22.064 |         3.66196 |
| 2024-01-02 00:00:00+00:00 |        70.308 |        11.669   |
| 2024-01-03 00:00:00+00:00 |        53.101 |         8.81317 |
| 2024-01-04 00:00:00+00:00 |        51.451 |         8.53932 |
| 2024-01-05 00:00:00+00:00 |        57.207 |         9.49465 |

# 📊 Visualizing Daily Energy Consumption and Cost with Dual Y-Axis Plot
This section uses matplotlib to create a dual-axis line chart that visualizes daily energy consumption (kWh) and cost ($) on the same plot, but with separate Y-axes for clarity.
This uses a twin axis plot so you can display both energy consumption and cost over time without overcrowding one axis. `fig.autofmt_xdate()` automatically rotates the X-axis date labels to prevent overlap.
The combined legend includes both data series, making the chart easy to interpret.
```js
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot daily energy consumption
color1 = 'tab:blue'
ax1.set_xlabel("Date")
ax1.set_ylabel("Consumption (kWh)", color=color1)
ax1.plot(daily_data.index, daily_data['consumption'], label='Consumption (kWh)', color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Second y-axis for cost
ax2 = ax1.twinx()
color2 = 'tab:purple'
ax2.set_ylabel("Cost ($)", color=color2)
ax2.plot(daily_data.index, daily_data['provided_cost'], label='Cost ($)', color=color2, linestyle='dashed', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Grid, legend, title
fig.suptitle("Daily Energy Consumption & Cost (Resampled from Hourly)", fontsize=16)
fig.autofmt_xdate()  # rotates date labels

# Combined legend from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('/kaggle/working/daily_energy_consumption.png', dpi=300)  # Save the plot
plt.show()
```
![image2](/Image/image2.png)

## 📌We can see there is a energy spike around 2024-01, and we need to figure out the reason that caused energy spike
# Applying Image understadning

```js
import PIL
prompt = [
  """
  Please give me a friendly explanation of what this image shows in a couple of paragraphs
  
  Please summarize key insights, including:
    - Typical daily usage and cost
    - Any significant variations or outliers such as around 2024-01 it has a consumption spike, find out the reasons might be
    - Suggestions to reduce energy waste or improve efficiency (even if rate is fixed)
    - whats the fixed-rate of per kWh?
  
  """,
  PIL.Image.open("/kaggle/working/daily_energy_consumption.png"),
]

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt
)
Markdown(response.text)
```
 ```JS
🤖:Significant Variations/Outliers: At the very beginning of January 2024 (around 2024-01), there is a massive spike in both consumption and cost, reaching over 200 kWh and a cost of over $35. Possible reasons for this spike could be:

Extreme Cold Weather: A sudden cold snap in early January could have caused a huge increase in heating demand (especially if you use electric heating).

Holiday Lighting: If you had elaborate holiday lights up, especially if left on for extended periods, that could contribute.

Guest(s): Maybe you had guests using more electricity to cook, wash cloths, etc.
Malfunctioning Appliance: An appliance, especially a heater or water heater, could have malfunctioned and run continuously.
```


## 🧪 Code Highlights

Optional: Paste a short code snippet or query example here.

```python
monthly_data = df.resample('M').agg({
    'consumption': 'sum',
    'provided_cost': 'sum'
}).round(2)
```

## 🔮 What's Next?

Describe possible extensions, future directions, or limitations:
- What would you build next?
- Any other data sources you wish you had?
- What challenges did you face?
