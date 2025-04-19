---
title: Watt-Seer Household  
layout: default
---

# 🌱 Watt-Seer Household  
**Smarter Homes, Greener Future**  
_GenAI Capstone Project by WattWise Innovators_

## 👥 Team Members 

- [Arushi Tariyal](https://www.kaggle.com/arushitariyal)  
- [Eric H. Adjakossa](https://www.kaggle.com/ericadjakossa)  
- [Lan H. Nguyen](https://www.kaggle.com/lannguyenrs)

---

## 🔍 Problem 

Ever stared at your energy bill wondering, **_“What’s using all that power?”_**  
That curiosity sparked our GenAI capstone. In a world of smart homes, why is understanding energy use still so hard?

🚨 **Why does this matter?**  
Because we live in a world of smart homes and connected devices, yet understanding home energy usage is still far from intuitive. Dashboards and charts exist—but they can feel impersonal, complex, or even intimidating to the average person.

What if, instead of scrolling through bar charts, you could just ask your home a question like,
**_"Why was my energy usage so high last winter?"_**
…and get an answer that makes sense?

🌍 **That was our mission:** to build a system where energy analytics becomes a conversation, not just a spreadsheet. And to do that, we combined efficient Artficial Intelligence, Machine Learning with the Natural Language power of Generative AI for Data Insights.

---

## 🤖 Solution  

We combined optimal Artificial Intelligence and Machine Learning, alongside Data Processing with the conversational power of GenAI.

- 🏠 Dataset: **500 smart homes** over one year  
- 📊 Inputs: Appliance usage, temperature, household size, and more  
- 🛠️ Techniques:  
    - 🧹 **Data profiling & cleaning:** Handled variable collection windows, missing values, and schema consistency  
    - 🧮 **Statistical analysis:** D’Agostino–Pearson normality test and Kruskal–Wallis H‑test to compare seasonal temperature distributions  
    - 📈 **Exploratory visualizations:** Seaborn relplots to reveal trends by season, appliance type, and household size  
    - 🤖 **Generative AI Q&A:** Gemini function calling with TF‑IDF fallback for real‑time, natural‑language responses  
- 💾 Data Type: **Structured data, having table in csv format**

Users ask plain English questions; AI runs the appropriate code and replies in conversational language.

---

## 🗝️ From Numbers to Meaning: Modeling & Analysis

- 🔍 **Normality Check:** D’Agostino–Pearson test showed outdoor temperature data is non‑Gaussian.  
- 🎯 **Seasonal Comparison:** Kruskal–Wallis H‑test revealed no significant differences in temperature distributions across seasons.  
- 📈 **Trend Visualizations:** Seaborn relplots confirmed that seasonal shifts have minimal impact on temperature patterns, and that air conditioners and heaters drive the highest energy use regardless of outdoor conditions or household size.  
- ➗ **Mean Aggregation:** Plotting mean energy consumption vs. temperature and household size further validated AC/heaters as peak loads, with fridges consistently at the low end.

---

## 🧠 GenAI Tech Stack

- 🦾 **Function Calling:** Executes exact code snippets for each user query.

- 🪄 **Structured Output:** Returns results in consistent, parseable formats.

- 🧩 **Embeddings + Vector Search:** Maps varied phrasing to the right data context.

- 🧲 **Retrieval-Augmented Generation (RAG):** Fetches the needed data before generating explanations.

- 📜 **Document Understanding:** Interprets dataset schema to maintain accuracy and relevance.

---

## 🎥 Project Walkthrough: Watch Our Energy‑Saving Innovation in Action 🌍

[![Watch the video](https://img.youtube.com/vi/Mrld6CWXUtg/hqdefault.jpg)](https://youtu.be/Mrld6CWXUtg)

💚 Click the image above and *Discover how “Watt Seer Household” empowers families to track their energy usage, reduce costs, and contribute to a greener planet — all with the help of AI.* 💕

---

## 💻 Try Watt-Seer-Household Yourself!
🔗 To view the Watt-Seer-Household Kaggle Notebook 👉 [click here!](https://www.kaggle.com/code/arushitariyal/watt-seer-household)

---
## 📊 Visualization Output: The Overall Energy Story, Visualized & Explained
🌠 **Imagine you input your Home IDs and want to acquire information based on it. Watt‑Seer‑Household gets to work and presents:**

### 🍂 1. Seasonal Temperature Trends
Here we see how outdoor temperature varies by season for your two home IDs.

![image](https://github.com/user-attachments/assets/1261e87c-5817-4ac7-abc1-a735cf7aa854)

*Caption: Outdoor Temperature (°C) over time for Home IDs 3 (left) and 4 (right)*

### ❄️ 2. Energy vs. Outdoor Temperature
Next, appliance‐level consumption plotted against outdoor temperature.

![image](https://github.com/user-attachments/assets/20f0c64a-9eae-40e7-bbeb-5629510d61a0)

*Caption: Energy Consumption (kWh) vs. Outdoor Temperature (°C) for Home IDs 1 (left) and 2 (right)*

### 🏡 3. Energy vs. Household Size
Same scatter, but now versus household size instead of temperature.

![image](https://github.com/user-attachments/assets/d09191f7-ee4c-4427-874f-fdb111669c25)

*Caption: Energy Consumption (kWh) vs. Household Size for Home IDs 1 (left) and 2 (right)*

---
🌻 **What if you want to see aggregated consumption? Watt-Seer-Household also illustrates the overall enery consumption based on different parameters:**

### ☀️ 1. Mean Energy vs. Mean Outdoor Temperature
Aggregate view: each appliance’s **mean** consumption against **mean** outdoor temperature.

![image](https://github.com/user-attachments/assets/4d354d0d-3ad8-451f-8694-f0b5f74773f4)

*Caption: Mean Energy Consumption (kWh) vs. Mean Outdoor Temperature (°C) across all homes*

### 🌇 2. Mean Energy vs. Mean Household Size
And finally, mean consumption versus mean household size.

![image](https://github.com/user-attachments/assets/6a5a2251-947b-442a-9bfd-f3e208a7b08b)

*Caption: Mean Energy Consumption (kWh) vs. Mean Household Size across all homes*

---

## ✨ The AI Analyst’s Take

📍 **LLM Comparison Summary:**  
> - 🌈 **Seasonal Temperature Trends:** Across peer homes, outdoor temperature profiles follow the same seasonal arcs—with consumption rising steadily in summer and dipping in winter—indicating shared climate-driven patterns rather than idiosyncratic behavior.  
> - 🌦️ **Energy vs. Outdoor Temperature:** In the scatter plots, air‑conditioning loads climb sharply once temperatures exceed ~25 °C, while heater usage dominates below ~10 °C; core appliances like fridge and lights stay tightly clustered regardless of temperature.  
> - 🏘️ **Energy vs. Household Size:** Per‑occupant energy use remains remarkably consistent across homes, with most of the between‑home variation driven by secondary loads (dishwasher, washing machine) rather than baseline appliances.

📌 **Aggregate Insights from Mean Trends:**  
> - 🌡️ **Temperature Sensitivity:** The **Mean Energy vs. Mean Outdoor Temperature** chart shows AC consumption rising at ~0.5 kWh/°C between 22–30 °C, while fridge and lighting vary by <0.05 kWh across the full range.  
> - 🔢 **Household Scaling:** The **Mean Energy vs. Mean Household Size** plot reveals dishwashers and washing machines each add about 0.2 kWh per extra occupant, whereas fridge and lights remain nearly flat (<0.05 kWh/member).  
> - 💡 **Efficiency Opportunities:** Many homes run their dishwasher ~10–15 % below the aggregate mean for their size—highlighting off‑peak scheduling as a simple way to shave peak demand.

By pairing **peer‑level comparisons** with **population‑level trends**, Watt‑Seer‑Compare turns raw kWh data into clear narratives and actionable insights—for example, pinpointing which appliance behaviors to target for greener, smarter living.  

---

## 🚀 Code Highlights Walkthrough

### 🧪 1. Statistical Validation: Normality & Kruskal–Wallis Seasonal Tests

```python
from scipy.stats import normaltest, kruskal

# 1. Normality test
_, p = normaltest(df['Outdoor Temperature (°C)'])
alpha = 0.05
if p > alpha:
    print("Outdoor temperatures look Gaussian.")
else:
    print("Outdoor temperatures don't look Gaussian.")

# 2. Kruskal–Wallis H‑test across seasons
_, p = kruskal(
    *[group['Outdoor Temperature (°C)'].values
      for _, group in df.groupby('Season')]
)
if p > alpha:
    print("Outdoor temperatures have the same distribution per season.")
else:
    print("Outdoor temperatures don't have the same distribution per season.")

# followed by correlation analysis & outlier detection
```
### ⏳ 2. Data Loading & Gemini Configuration

```python
import pandas as pd
import numpy as np
from kaggle_secrets import UserSecretsClient
from google import generativeai as genai

df = pd.read_csv("/kaggle/input/.../smart_home_energy_consumption_large.csv")
df.columns = df.columns.str.strip()

api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# followed by preliminary EDA, datetime parsing, data validation routines
```
### 📚 3. TF‑IDF‑Based Chunk Retrieval

```python
from sklearn.feature_extraction.text import TfidfVectorizer

dataset_text = df.to_csv(index=False)
def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(dataset_text)
vectorizer = TfidfVectorizer().fit(chunks)
chunk_embeddings = vectorizer.transform(chunks)

def retrieve_relevant_chunks(query, top_k=1000):
    query_vec = vectorizer.transform([query])
    similarities = (chunk_embeddings * query_vec.T).toarray().flatten()
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]
```
### 🔁 4. Prompt Construction & Retry Logic

```python
import time

def build_prompt(query):
    context = "\n".join(retrieve_relevant_chunks(query))
    return f"{context}\n\nQuestion: {query}\n\nAnswer:"

def gemini_answer(prompt):
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "quota" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                return f"Gemini API error: {e}"
    return "Gemini failed after retries." 
```
### 🌡️ 5. Energy‑by‑Parameter Visualization Helper

```python
def plot_energy_by_appliance():
total = df.groupby("Appliance Type")["Energy Consumption (kWh)"] \
          .sum().sort_values()
total.plot(kind="barh", figsize=(10, 6))
plt.title("Total Energy Consumption by Appliance")
plt.xlabel("Energy Consumption (kWh)")
plt.show()

# followed by plot_monthly_usage & plot_household_size
```
### 🐼 6. Pandas + GenAI Query Handler
```python
def answer_query(query):
    q = query.lower().strip().replace("usage", "energy consumption")
    if "temperature" in q and "energy" in q:
        df_clean = df[["Outdoor Temperature (°C)", "Energy Consumption (kWh)"]].dropna()
        df_clean["Temp Range"] = pd.cut(
            df_clean["Outdoor Temperature (°C)"],
            bins=[-20, 0, 10, 20, 30, 50],
            labels=["Freezing", "Cold", "Cool", "Warm", "Hot"]
        )
        avg_usage = df_clean.groupby("Temp Range", observed=True)["Energy Consumption (kWh)"].mean()
        pandas_answer = "📊 Average energy consumption by temperature range:\n\n" + avg_usage.to_string()
        prompt = (
            f"You are a helpful data analyst assistant.\n"
            f"Pandas Result: {pandas_answer}\n"
            f"Question: {query}\n"
            f"Explanation:"
        )
        return f"{pandas_answer}\n\nGemini Explanation: {gemini_answer(prompt)}"
    # …other patterns…
    return gemini_answer(build_prompt(query))

    # followed by other patterns, fallback, visuals
```
### ▶️ 7. REPL‑Style Main Loop

```python
def main():
    print("Enter a query (type 'quit' to stop or 'visuals' to view charts):")
    while True:
        query = input("Question: ").strip()
        if query.lower() == "quit":
            print("Exiting. Goodbye!")
            break
        elif "visual" in query.lower():
            plot_energy_by_appliance()
            plot_monthly_usage()
            plot_household_size()
        else:
            ans = answer_query(query)
            print(f"\n❓Q: {query}\n💡A: {ans}\n")

main()
```

---

## 🔮 Code Output

> **Q:** Which types of appliances used the most energy?  
> **A:** Top 5 Appliances by Energy Consumption:


| Appliance Type       | Total Energy (kWh) |
|----------------------|-------------------:|
| Air Conditioning     | 35,233.06         |
| Heater               | 34,930.78         |
| Dishwasher           | 11,138.51         |
| Lights               | 11,092.12         |
| Oven                 | 10,963.51         |


💡 _Air Conditioning and Heater consumed the most energy, significantly outpacing other appliances._

----

> **Q:** What was the total energy consumption in January?  
> **A:** The total consumption in January was **3,234.75 kWh**.


| Month   | Total Energy (kWh) |
|---------|-------------------:|
| January | 3,234.75           |


💡 _January shows moderate energy usage, likely due to heating and lighting needs during colder months._

---



## 🎬 What's Next?

🔭 Scale this to real smart homes or partner utilities:

- 🔔 Personalized alerts like:  
  _“Your heating use was 20% above average. Lowering by 1°C can save $50.”_
- 📆 Forecasting next month’s energy bill
- 🗣️ Voice assistant integration
- 📶 Connect to IoT devices for live feedback

---

## 💬 Final Thoughts

This project was about giving **data a voice** with conversational AI can transform raw energy logs into clear, actionable guidance. By giving your data a human voice, Watt‑Seer_Household empowers homeowners to uncover insights, drive efficiency, and make greener choices—one question at a time.

🚀 _Smarter homes. Better answers. Greener future._ 💎
