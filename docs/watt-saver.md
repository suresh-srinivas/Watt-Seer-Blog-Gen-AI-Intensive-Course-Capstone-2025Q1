
---
title: Watt-Saver Project Template
layout: default
---

# 🧠 Watt-Saver the Personalized Energy Coach

This is the blog for [Gen AI Intensive Course Capstone 2025Q1 Porject](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1) which is part of [5-Day Gen AI Intensive Course with Google](https://rsvp.withgoogle.com/events/google-generative-ai-intensive_2025q1)


💡 Understanding Energy Bills with GenAI:

Many people receive energy bills without knowing why they’re high or how to reduce them. This project shows how Generative AI (GenAI) can help users understand their energy consumption and get personalized, human-readable feedback.

❓ Why/How GenAI can solve the problem?

Charts and numbers alone don’t help most people take action. GenAI transforms raw energy data into insightful narratives—like identifying your most expensive usage days and suggesting practical changes in behavior or device usage.

🔍 How It Works
We used hourly energy consumption data from a single home. After cleaning and summarizing it, we prompted Gemini (a GenAI model) to explain trends—like why usage spiked on certain days—and offer tips to reduce costs.

📊 Key Results
Instead of just visualizing data, Gemini responded with plain-language insights and personalized advice. It helped reveal things like:
- Which time of day is most expensive
- When heating/cooling is overused
- How to shift usage to off-peak hours

🚀 Impact
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
