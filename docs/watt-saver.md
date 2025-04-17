
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

We can see the plot to have an quick grasp of the data and see Air conditioning and Heater have the highest energy consumption during the time period.
![prompt-analysis](/Image/prompt-analysis.png)



## Apply `Document understanding` to describe the datasets
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
