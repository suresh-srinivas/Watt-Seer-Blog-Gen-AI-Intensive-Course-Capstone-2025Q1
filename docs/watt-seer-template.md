
---
title: Watt-Seer Project Template
layout: default
---

# 🧠 Project Title

Replace this title with your project name, e.g. "Watt-Seer Coach"

## 🔍 Problem

Briefly describe the specific problem you’re addressing.
- Why does this matter?
- What insight or change are you enabling with AI?

## 🤖 Solution

Explain what your project does using Kaggle + Gemini.
- What datasets or inputs do you use?
- What Gen AI techniques or prompts are applied?
- Are you using image, text, or structured data?

## 👥 Team Members

- Member 1
- Member 2
- Member 3

## 💻 Notebook Link

[🔗 View on Kaggle](#)  
(Replace the # with your actual notebook URL)

## 📊 Example Output

You can describe or paste a sample chart, graph, or summary result.

```
Sample Output:
January 16: 237 kWh used
Insight: Heat pump likely switched to resistance heating due to freezing temperatures
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
