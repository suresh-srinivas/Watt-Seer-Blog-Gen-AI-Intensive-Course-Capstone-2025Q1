
---
title: "Watt-Seer: How Three Neighbors Used AI to Understand Their Energy Usage"
description: A storytelling case study on using Kaggle + Gemini Vision to compare home energy data and extract insights from scanned bills.
date: 2024-04-16
author: Suresh Srinivas
layout: post
---

# 🔋 Watt-Seer: How Three Neighbors Used AI to Understand Their Energy Usage

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

## 🛠️ What They Built: Watt-Seer

- Ed provided the **hourly energy consumption data** from his utility.
- Jerry gave Anita a **photo of his electric bill**.
- Anita spun up a Kaggle notebook, loaded Pandas, and brought in Gemini Vision.

### The Result:

✅ Monthly summaries from Ed's data  
✅ Extraction of key details (kWh, dates, cost) from Jerry's scanned bill  
✅ AI-generated comparisons and suggestions for energy-saving actions

---

## 📉 Ed's Consumption on a Cold Week

> "You used 237 kWh on January 16 alone," Anita pointed out. "Looks like your heat pump switched to resistance mode during the cold snap."

---

## 🧾 Jerry's Bill, Extracted by AI

Gemini Vision read Jerry's scanned bill and returned:

- Billing period: Jan 5 to Feb 5  
- Usage: 3,200 kWh  
- Cost: $386  
- Estimated rate: $0.12/kWh

---

## 💬 Gemini-Powered Recommendations

> "Jerry's home may be better insulated or use a gas furnace. Ed's electric resistance heating likely caused the winter spike. Consider supplemental heating or sealing air leaks."

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
monthly_text = monthly_data.to_string()

prompt = f'''
This is my neighbor's electric bill image. Please extract monthly kWh usage.

Here's my usage for the same months:
{monthly_text}

Compare our energy profiles and explain the differences.
'''
```

### 🔹 Sending a bill image to Gemini Vision

```python
with open("neighbors_bill.jpg", "rb") as img_file:
    image_data = img_file.read()

response = client.models.generate_content(
    model='gemini-pro-vision',
    contents=[
        {"text": prompt},
        {"image": image_data}
    ]
)
```

---

## 🚧 Limitations & What's Next

- 📷 **Image extraction isn't perfect**: Gemini Vision might misread handwritten notes or low-res scans.
- 💬 **AI-generated insights may need validation**: Always double-check suggestions with a professional (especially for HVAC or insulation upgrades).
- 🔮 **Future Possibility**: Automatically pull utility bills via API, integrate smart thermostat data, or use fine-tuned models for your home's energy profile.

---

## 🤝 From Personal Curiosity to Community Action

By combining code, AI, and community, Ed, Jerry, and Anita turned a conversation into insight.

If you've got a folder of bills — or a zip of usage data — maybe it's your turn next.

---

## 🔗 Try Watt-Seer Yourself

👉 [Link to the Kaggle Notebook](#)  
*(Replace this with the actual link to your notebook)*

Upload your own usage data. Or just bring a photo of your bill.  
Let the AI do the explaining.  
You've got energy stories waiting to be told.
