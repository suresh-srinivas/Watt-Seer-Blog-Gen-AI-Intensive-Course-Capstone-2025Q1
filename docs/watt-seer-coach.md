---
title: "Watt-Seer: How Three Neighbors Used AI to Understand Their Energy Usage"
layout: default
---
# Watt-Seer - Personalized Energy Coach
## This is the blog for [Gen AI Intensive Course Capstone 2025Q1 Porject](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1) which is part of [5-Day Gen AI Intensive Course with Google](https://rsvp.withgoogle.com/events/google-generative-ai-intensive_2025q1)

 A storytelling case study on using Kaggle + Gemini to compare home energy data and extract insights from scanned bills.

## 👥 Team Members

- [Ashwini Apte](https://www.kaggle.com/ashwiniapte)
- [Suresh Srinivas](https://www.kaggle.com/sureshsrinivas)
- [Rao Parasa](https://www.kaggle.com/raoparasa) 

# 
# 🔋 Watt-Seer - Personalized Energy Coach: How Three Neighbors Used AI to Understand Their Energy Usage

In a quiet cul-de-sac in Portland, three retired neighbors found themselves in a uniquely 21st-century situation: they wanted to understand their electric bills — and only one of them knew how to code.

## 👨‍🔧 The Engineer and the Salesman

**Ed**, a retired electrical engineer, is the neighborhood's unofficial handyman. His garage is filled with sensors, solar panels, and spreadsheets. One day in January, he noticed his electricity bill had doubled. His reaction?  
> "I downloaded my entire year's hourly usage data from Portland General Electric and wrote a script to find the peak days."

**Jerry**, the neighbor across the street, used to be a salesman. He's old-school — keeps all his electric bills in a manila folder. When he heard Ed talking about kilowatt-hours and usage curves, he just shook his head.  
> "I've got the bill right here," he said, waving a paper copy. "But what does it all mean?"

## 🤖 Enter Anita: The AI Neighbor

**Anita**, the third neighbor, had just left her role running a boutique AI consultancy. She overheard the discussion on one of her dog walks.  
> "You know," she smiled, "you two are sitting on a goldmine of data. Want help turning it into something useful?"

Together, they launched a weekend project to build something simple, visual, and smart: **Watt-Seer** — an AI-powered notebook that turns **raw energy data and scanned bills** into personalized energy insights.

---

## 🔍 The Problem: Understanding Home Energy Use

Most people don't understand what's driving their energy bills. Even when they have access to hourly or daily usage, it's just rows of numbers.

On the other hand, millions of Americans only have **paper bills** with **monthly totals**, and no tools to compare or analyze.

**Gen AI, especially multimodal models like Gemini, can bridge this gap — turning structured data and unstructured images into meaning.**

---

## 🛠️ What They Built: Watt-Seer Personalization Coach

- Ed provided the **hourly energy consumption data** from his utility.
- Jerry gave Anita a **photo of his electric bill**.
- Anita spun up a Kaggle notebook, loaded Pandas, and brought in Gemini Vision.

### The Result:

✅ Monthly summaries from Ed's data  
✅ Extraction of key details (kWh, dates, cost) from Jerry's scanned bill  
✅ AI-generated comparisons and suggestions for energy-saving actions

---

## 📉 Ed's Consumption on a Cold Week

![Energy Usage Graph](images/energy-graph.png)

> "You used 237 kWh on January 16 alone," Anita pointed out. "AI root caused it and it Looks like your heat pump switched to resistance mode during the cold snap."

---

## 🧾 Jerry's Bill, Extracted by AI

Gemini Vision read Jerry's scanned bill and returned:

- Billing period: Jan 5 to Feb 5  
- Usage: 3,200 kWh  
- Cost: $528  
- Estimated rate: $0.165/kWh

---

## 💬 Gemini-Powered Recommendations

> " Ed's electric resistance heating likely caused the winter spike. Consider supplemental heating or sealing air leaks."

---

## 🧪 Sample Code from the Watt-Seer Notebook

### 🔹 Resampling hourly data to monthly

```python
df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
df = df.set_index('start_time')

monthly_data = df.resample('M').agg({
    'consumption': 'sum',
    'provided_cost': 'sum'
}).round(2)
```

### 🔹 Formatting for Gemini comparison

```python
compare_prompt = f"""
Here is my neighbor’s extracted monthly energy usage:

{neighbor_usage_summary}

Here is my own energy usage during the same months:

{monthly_text}

Please compare our energy usage and suggest why there might be differences. We both are in Portland, OR and are a two person household. Use a simple ratio of my usage/neighbors. Also mention whether the usage levels are typical for similar homes.
"""

response_compare = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=compare_prompt
)

Markdown(response_compare.text)

```

### 🔹 Sending a bill image to Gemini

```python
prompt = [
  f"This is my neighbor’s electric bill. Please extract the monthly energy usage (in kWh) for all the months",
  PIL.Image.open("/kaggle/input/neighborbill/Neighbor-Bill.jpeg")
]

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt
)
neighbors_energy = response.text
Markdown(response.text)
```

---

## 🚧 Limitations & What's Next

- 📷 **Image extraction isn't perfect**: Gemini misread handwritten notes or low-res scans.
- 💬 **AI-generated insights may need validation**: Always double-check suggestions with a professional (especially for HVAC or insulation upgrades).
- 🔮 **Future Possibility**: Automatically pull utility data through API, integrate other energy data (eg solar) and include smart thermostat data to tell the house temperature, or use fine-tuned models for your home's energy profile.

---

## 🤝 From Personal Curiosity to Community Action

By combining code, AI, and community, Ed, Jerry, and Anita turned a conversation into insight.

If you've got a folder of bills — or a zip of usage data — maybe it's your turn next.

---

## 🔗 Try Watt-Seer Yourself


[🔗 View on Kaggle](https://www.kaggle.com/code/sureshsrinivas/watt-seer-personalized-energy-coach)  

Upload your own usage data. Or just bring a photo of your bill.  
Let the AI do the explaining.  
You've got energy stories waiting to be told.
