# 💡 Watt-Seer-Compare: Is Your Energy Bill "Normal"? AI Finds Your Household's 'Energy Twins'

You stare at your energy bill. The numbers fluctuate month to month, but one question always lingers: **"Is this *normal*?"**  You check the usage – hundreds, maybe thousands of kilowatt-hours (kWh). But what does that really mean? Compared to whom? Is your home an energy guzzler, a model of efficiency, or just... average?

It's a question millions ask, often met with frustratingly little context. Raw smart meter data is overwhelming, and simple monthly totals hide the real story. What if you could see how your home truly stacks up against others just like it? Maybe you live in a neighborhood with similar houses, but you don't know your neighbors' habits. Maybe the weather was weird last month. How does your specific energy usage pattern – driven by your appliances, your schedule, and your household size – compare to others like you?

That's the quest behind Watt-Seer-Compare. We're moving beyond simple averages to provide personalized, AI-powered energy benchmarking. Using detailed smart home data and the nuanced understanding of Google Gemini, this project finds your household's "Energy Twins" and tells you why your usage patterns are similar or different.

## 🔍 The Problem: Drowning in Data, Starving for Context

Smart meters promise granular insights, tracking energy use down to the hour or minute. We get data points enriched with details:   

- 🔌 Appliance Type: Fridge, Heater, AC, Oven...
- ⚡ Energy Consumption (kWh): The precise usage.
- 📅 Date + Time: When energy was used.
- 🌡️ Outdoor Temperature: A key driver for heating/cooling.   
- ☀️ Season: Context for usage patterns.
- 👨‍👩‍👧‍👦 Household Size: A major factor in demand.   

But rows upon rows of this data often lead to more confusion than clarity. How do you know if your AC usage on a hot August day was typical for a family of four in your climate zone? 

Traditional comparisons often rely on crude filters (e.g., same month), missing the subtle interplay of factors that define a household's energy signature.

![image](https://github.com/user-attachments/assets/a76dde97-9931-4739-83cb-0b6fdb9cf19e)

## 🤖 Solution: Finding Your "Energy Twins" with AI

**Watt-Seer-Compare** tackles this challenge head-on. It doesn't just filter data; it understands it. Here’s how:

Smart Summaries: We first distill the flood of raw data into concise, informative monthly summaries for each home, capturing total usage, temperature context, seasonality, household size, and key appliance contributions.

Semantic Understanding (Gemini Embeddings): This is where the magic happens. We use Google Gemini's text-embedding-004 model to convert these textual summaries into rich numerical representations (embeddings). These aren't just numbers; they capture the semantic meaning and context of each household's monthly energy story.

Finding Your Peers (Cosine Similarity): With embeddings, we can mathematically compare how "alike" different households are based on their entire energy profile for the month using cosine similarity. This allows us to find the homes that are most similar to yours – your "Energy Twins."

Visual Comparisons: We generate clear, intuitive plots:
Your home's appliance breakdown (pie chart).
Your total usage vs. the average of your Energy Twins (bar chart).
Your total usage vs. each individual Energy Twin (bar chart).
Your top appliance usage vs. the average of your Twins (grouped bar chart).

AI Explanation (Gemini LLM): Numbers and plots are great, but why are you similar or different? We feed your summary and your Twins' summaries into Gemini (gemini-2.0-flash), prompting it to analyze the nuances and generate a plain-English explanation comparing your profile to your closest peers.

&lt;center>
[Diagram: A flowchart illustrating the process: Raw Data -> Monthly Summary -> Gemini Embedding -> Cosine Similarity -> Find "Energy Twins" -> Generate Plots -> Gemini LLM Explanation -> User Insight.]
&lt;/center>

## 📊 Example Output: Your Energy Story, Visualized & Explained
Imagine you input your Home ID (say, 66) and the month (2023-08). Watt-Seer-Compare gets to work and presents:

1. Your Home's Profile:

<img width="619" alt="Screenshot 2025-04-17 at 5 20 17 PM" src="https://github.com/user-attachments/assets/5d1d945d-1c70-4eef-9c8f-dc48a075de05" />

Caption: Target Home 66 - Appliance Energy Breakdown (2023-08)

2. Comparison Context:

<img width="503" alt="Screenshot 2025-04-17 at 5 20 30 PM" src="https://github.com/user-attachments/assets/039d188e-5b03-4139-97e6-741d187b9f96" />

Caption: Total Monthly Energy Comparison (2023-08) - Target vs. Average Similar

<img width="698" alt="Screenshot 2025-04-17 at 5 20 40 PM" src="https://github.com/user-attachments/assets/fcdcb405-c3d7-4bdd-8541-e4a98658b401" />

Caption: Target vs. Individual Similar Homes - Total Monthly Energy (2023-08)

<img width="788" alt="Screenshot 2025-04-17 at 5 21 11 PM" src="https://github.com/user-attachments/assets/7641a98b-ce55-4c22-ac90-cf736dc04e91" />

Caption: Comparison of Top Appliance Usage (2023-08) - Target vs. Average Similar

3. The AI Analyst's Take:

 LLM Comparison Summary:

Home 66 exhibits an energy consumption profile in August 2023 that is highly similar to homes 46, 131, and 124, primarily due to comparable total energy usage (around 30-33 kWh), similar seasonal context (Summer), and the significant contribution of air conditioning... While the household size is comparable across the homes, some key differences exist. Home 46 has a slightly larger household. Homes 46 and 131 also report heater usage, while Home 124's energy consumption is driven more by microwave and computer usage than air conditioning...

Suddenly, your energy usage isn't just a number; it's a narrative placed within the context of genuinely comparable households.

## 🧪 Code Highlight: Generating the Embeddings

The core of finding "Energy Twins" lies in creating meaningful embeddings from the monthly summaries. Here's a snippet of how we prepare the data and call the embedding model (using helper functions defined in the notebook):

**1. Generating Monthly Text Summaries:**

```python
# (Inside create_monthly_summaries function)
summary_text = (
    f"Home ID {home_id} for Month {period}:\n"
    f"- Total Consumption: {total_kwh:.1f} kWh\n"
    f"- Household Size: {household_size}\n"
    f"- Average Temperature: {avg_temp}°C\n"
    f"- Season: {season}\n"
    f"- Appliance Breakdown: {appliance_summary}"
)
summary_records.append({..., "Summary Text": summary_text})
```
**2. Finding Similar Homes via Embeddings:**

```python
# (Inside run_energy_comparison_analysis)
# Get embeddings for the month
embeddings, summaries = generate_embeddings_efficiently(...)

# Calculate similarity
top_similar, bottom_similar = get_embedding_similarities(
    target_home_id=target_home_id,
    all_embeddings=embeddings,
    k=k_similar
)
```
**3. Prompting the LLM for Explanation:**

```python
# (Inside generate_llm_comparison)
prompt = f"""
**Context:** You are an expert energy analyst...
**Input Data:**
* **Target Home (ID: {target_home_id}) Summary:**\n    ```{target_summary}```
* **Most Similar Homes (Top {homes_added}):**\n{similar_homes_text}
**Task:** Analyze the provided summaries. **Think step-by-step**...
**Output:** Provide the final synthesized comparison summary...
"""
response = client.models.generate_content(model=model_name, contents=prompt, ...)
```
## ✨ Why This Matters: Beyond Simple Filters
### This approach offers significant advantages:

Nuanced Similarity: Embeddings capture complex relationships between factors (temperature, appliance use, household size) that simple filters miss. An "Energy Twin" found this way shares a more holistic usage pattern.   

Contextual Explanations: The LLM doesn't just state facts; it synthesizes information from multiple summaries to explain why similarities or differences exist, providing actionable context.

Data-Driven Discovery: You might find your home is similar to others for unexpected reasons, uncovering potential inefficiencies or validating your energy-saving efforts.

## 🔮 Limitations & Future Directions..

Watt-Seer-Compare provides powerful context, but like any analysis, it has boundaries:

Data Dependency: The quality and granularity of the input smart meter data are crucial.

Dataset Bias: Comparisons are only as good as the diversity of homes within the dataset.

Model Evolution: AI models change; prompts and interpretations may need adjustments over time.

What's Next?
Integrating real-time data streams.

Adding energy cost analysis and prediction.

Incorporating user feedback to refine similarity.

Generating personalized energy-saving recommendations based on comparison insights.

Expanding to use multimodal inputs (like scanned bills, similar to the original Watt-Seer concept).

## 🤝 From Numbers to Narratives: 
Understand Your Energy Story

Stop wondering in the dark. By finding your "Energy Twins", combining detailed data, semantic AI, visualizations, and natural language explanations, Watt-Seer-Compare demonstrates how AI can transform rows of energy data into a meaningful conversation about consumption patterns and empowers you to understand your unique energy story.

Ready to see how you stack up?

## 🔗 Try Watt-Seer-Compare Yourself!

👉 View the Watt-Seer-Compare Kaggle Notebook
(Replace # [Link-To-Your-Kaggle-Notebook] with the actual public URL)

Explore the code, run it with the sample data, and imagine plugging in your own smart meter readings. Discover your own "Energy Twins" and gain clarity on your consumption.

Your energy data holds insights waiting to be revealed. Let AI help you listen.


Sources and related content
