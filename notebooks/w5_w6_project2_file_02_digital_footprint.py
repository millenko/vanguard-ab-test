#!/usr/bin/env python
# coding: utf-8

# # 3. Success indicators
# - **3.1. Test group, 3.2. Control group:**
#   - **3.X.1. Completion rate:** The proportion of users who reach the final ‘confirm’ step.
#   - **3.X.2. Time spent on each step:** The average duration users spend on each step.
#   - **3.X.3. Error rates:** Steps where users go back to a previous step.
# - **3.3. Redesign outcome:** Given the 3 KPIs, how the new design’s performance compare to the old one?
# 
# # 4. Hypotheses testing
# - **4.1. Completion rate:** z-test
# - **4.2. Error rate:** z-test
# - **4.3. various means of attrbutes:** t-test
# 
# # 5. Experiment evaluation

# In[ ]:


import pandas as pd
import numpy as np
from IPython.display import display, HTML
import scipy.stats as st
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_client_profiles = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/final/df_client_profiles_final.csv")
df_digital_footprint = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/final/df_digital_footprint_final.csv")


# # Dataset exploration

# In[ ]:


df_client_profiles.head()


# In[ ]:


# Drop the 'Dummy' column from df_digital_footprint
df_client_profiles = df_client_profiles.drop(columns=['dummy'])


# In[ ]:


df_digital_footprint.describe(include="all")


# In[ ]:


df_digital_footprint.info()


# In[ ]:


# Cast "date_time" as datetime type.
df_digital_footprint["date_time"] = pd.to_datetime(df_digital_footprint["date_time"])


# In[ ]:


# Inner-joining (df_digital_footprint, df_client_profiles) to only keep the rows with known user_id.
df_digital_footprint = pd.merge(df_digital_footprint, df_client_profiles, on="client_id")
df_digital_footprint


# In[ ]:


# Drop duplicated rows. There are no null values.
print(f"pre-drop duplicated rows: {df_digital_footprint.duplicated().sum()}")
print(f"null values: {df_digital_footprint.isna().sum().sum()}")

df_digital_footprint = df_digital_footprint.drop_duplicates()
print(f"post-drop duplicated rows: {df_digital_footprint.duplicated().sum()}")


# # 3.1. Test group: success indicators

# ## 3.1.1. completion rate

# In[ ]:


df_test_group = df_digital_footprint[df_digital_footprint["experiment_group"] == "Test"]

confirm_count_test = (df_test_group["process_step"] == "confirm").sum()
print(f"Number of 'confirm' steps: {confirm_count_test}")

unique_visit_count_test = df_test_group["visit_id"].nunique()
print(f"Number of unique visit_id: {unique_visit_count_test}")

completion_rate_test = confirm_count_test / unique_visit_count_test
print(f"The proportion of users who reach the final ‘confirm’ step: {completion_rate_test:.2%}")


# ## 3.1.2. time spent on each step

# In[ ]:


test_durations = {}
test_total_duration = pd.Timedelta(0)

step_pairs = [
    ("start", "step_1"),
    ("step_1", "step_2"),
    ("step_2", "step_3"),
    ("step_3", "confirm"),
]

for start_step, end_step in step_pairs:
    
    relevant_steps = df_test_group[df_test_group["process_step"].isin([start_step, end_step])]
    
    relevant_steps_sorted = relevant_steps.sort_values(by=["visit_id", "date_time"])
    
    # Calculate the time difference between steps for each visit_id
    relevant_steps_sorted["time_diff"] = relevant_steps_sorted.groupby("visit_id")["date_time"].diff()
    
    # Select end_step rows to use the calculated time differences as durations from start_step to end_step
    end_step_durations = relevant_steps_sorted[relevant_steps_sorted["process_step"] == end_step]
    
    # Calculate the mean duration from start_step to end_step
    test_mean_duration = end_step_durations["time_diff"].mean()
    
    # Format the mean duration string to include minutes, and seconds
    test_mean_duration_str = f"{test_mean_duration.components.minutes}m {test_mean_duration.components.seconds}s"
    
    # Store the result in the dictionary keyed by the step pair
    test_durations[(start_step, end_step)] = test_mean_duration_str
    
    # Add the mean duration to the total duration
    test_total_duration += test_mean_duration

    print(f"{start_step} to {end_step}: {test_mean_duration_str}")

# Format the total duration string to include minutes, and seconds
test_total_duration_str = f"{test_total_duration.components.minutes}m {test_total_duration.components.seconds}s"

print(f"Total duration for the whole journey: {test_total_duration_str}")


# ## 3.1.3. error rates

# In[ ]:


# Sort by visit_id and date_time to ensure chronological order
df_test_group = df_test_group.sort_values(by=["client_id", "visit_id", "date_time"])

# Assign step orders
step_order = {"start": 0,
              "step_1": 1,
              "step_2": 2,
              "step_3": 3,
              "confirm": 4}

# Detect backward movements
df_test_group["step_order"] = df_test_group["process_step"].map(step_order)

# Calculate the difference in step order to identify backward movements
df_test_group["step_diff"] = df_test_group.groupby(["client_id", "visit_id"])["step_order"].diff()

# A negative step_diff indicates a backward movement
df_test_group["is_backward"] = df_test_group["step_diff"] < 0

# Proportion of sessions with at least one backward movement
error_sessions_test = df_test_group[df_test_group["is_backward"]].groupby(["client_id", "visit_id"]).ngroups
total_sessions_test = df_test_group.groupby(["client_id", "visit_id"]).ngroups
test_error_rate = error_sessions_test / total_sessions_test

print(f"Proportion of Test group sessions with errors: {test_error_rate:.2%}")

# Count of backward movements by step
errors_by_step = df_test_group[df_test_group["is_backward"]]["process_step"].value_counts()
errors_by_step


# # 3.2. Control group: success indicators

# ## 3.2.1. completion rate

# In[ ]:


df_control_group = df_digital_footprint[df_digital_footprint["experiment_group"] == "Control"]

confirm_count_control = (df_control_group["process_step"] == "confirm").sum()
print(f"Number of 'confirm' steps: {confirm_count_control}")

unique_visit_count_control = df_control_group["visit_id"].nunique()
print(f"Number of unique visit_id: {unique_visit_count_control}")

completion_rate_control = confirm_count_control / unique_visit_count_control
print(f"The proportion of users who reach the final 'confirm' step: {completion_rate_control:.2%}")


# ## 3.2.2. time spent on each step

# In[ ]:


control_durations = {}
control_total_duration = pd.Timedelta(0)

step_pairs = [
    ("start", "step_1"),
    ("step_1", "step_2"),
    ("step_2", "step_3"),
    ("step_3", "confirm"),
]

for start_step, end_step in step_pairs:
    
    relevant_steps = df_control_group[df_control_group["process_step"].isin([start_step, end_step])]
    
    relevant_steps_sorted = relevant_steps.sort_values(by=["visit_id", "date_time"])
    
    # Calculate the time difference between steps for each visit_id
    relevant_steps_sorted["time_diff"] = relevant_steps_sorted.groupby("visit_id")["date_time"].diff()
    
    # Select end_step rows to use the calculated time differences as durations from start_step to end_step
    end_step_durations = relevant_steps_sorted[relevant_steps_sorted["process_step"] == end_step]
    
    # Calculate the mean duration from start_step to end_step
    control_mean_duration = end_step_durations["time_diff"].mean()
    
    # Format the mean duration string to include minutes, and seconds
    control_mean_duration_str = f"{control_mean_duration.components.minutes}m {control_mean_duration.components.seconds}s"
    
    # Store the result in the dictionary keyed by the step pair
    control_durations[(start_step, end_step)] = control_mean_duration_str
    
    # Add the mean duration to the total duration
    control_total_duration += control_mean_duration

    print(f"{start_step} to {end_step}: {control_mean_duration_str}")

# Format the total duration string to include minutes, and seconds
control_total_duration_str = f"{control_total_duration.components.minutes}m {control_total_duration.components.seconds}s"

print(f"Total duration for the whole journey: {control_total_duration_str}")


# ## 3.2.3. error rates

# In[ ]:


# Sort by visit_id and date_time to ensure chronological order
df_control_group = df_control_group.sort_values(by=["client_id", "visit_id", "date_time"])

# Assign step orders
step_order = {"start": 0,
               "step_1": 1,
               "step_2": 2,
               "step_3": 3,
               "confirm": 4}

# Detect backward movements
df_control_group["step_order"] = df_control_group["process_step"].map(step_order)

# Calculate the difference in step order to identify backward movements
df_control_group["step_diff"] = df_control_group.groupby(["client_id", "visit_id"])["step_order"].diff()

# A negative step_diff indicates a backward movement
df_control_group["is_backward"] = df_control_group["step_diff"] < 0

# Proportion of sessions with at least one backward movement
error_sessions_control = df_control_group[df_control_group["is_backward"]].groupby(["client_id", "visit_id"]).ngroups
total_sessions_control = df_control_group.groupby(["client_id", "visit_id"]).ngroups
control_error_rate = error_sessions_control / total_sessions_control

print(f"Proportion of Control group sessions with errors: {control_error_rate:.2%}")

# Count of backward movements by step
errors_by_step = df_control_group[df_control_group["is_backward"]]["process_step"].value_counts()
errors_by_step


# # 3.3. Redesign outcome

# In[ ]:


# Test group has higher completion rate
print(f"The proportion of Test group reaching the final 'confirm' step: {completion_rate_test:.2%}")
print(f"The proportion of Control group reaching the 'confirm' step is: {completion_rate_control:.2%}\n")

# Control group has lower average time spent between steps
for start_step, end_step in step_pairs:
    test_str = test_durations.get((start_step, end_step), "No data")
    control_str = control_durations.get((start_step, end_step), "No data")
    print(f"Transition from '{start_step}' to '{end_step}':")
    print(f"  Test group: {test_str}")
    print(f"  Control g.: {control_str}\n")

# Control group has lower proportion of sessions with errors.
print(f"Proportion of Test group sessions with errors: {test_error_rate:.2%}")
print(f"Proportion of Control g. sessions with errors: {control_error_rate:.2%}")


# In[ ]:


completion_rates = {"test": completion_rate_test, "control": completion_rate_control}
groups = list(completion_rates.keys())
rates = [completion_rates[group] for group in groups]

plt.figure(figsize=(8, 6))
bar_positions = range(len(groups))
colors = ["darkorange", "royalblue"]
bar_width = 0.25
plt.bar(bar_positions, rates, width=bar_width, color=colors, align='center')

plt.xlabel("group")
plt.ylabel("completion rate")
plt.title("completion rates by group")
plt.xticks(bar_positions, groups)
plt.ylim(0, max(rates) + 0.05)

for i, rate in enumerate(rates):
    plt.text(i, rate, f"{rate:.2%}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("completion rate by group.png", dpi=300)
plt.show()


# In[ ]:


test_total_minutes = test_total_duration.total_seconds() / 60
control_total_minutes = control_total_duration.total_seconds() / 60

test_duration_str = f"{int(test_total_minutes)}m {int((test_total_minutes % 1) * 60)}s"
control_duration_str = f"{int(control_total_minutes)}m {int((control_total_minutes % 1) * 60)}s"

groups = ["test", "control"]
total_durations_minutes = [test_total_minutes, control_total_minutes]
colors = ["darkorange", "royalblue"]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(groups, total_durations_minutes, color=colors, width=0.25)

ax.set_ylabel("total duration (minutes)")
ax.set_title("total duration comparison between test and control groups")

for bar, duration_str in zip(bars, [test_duration_str, control_duration_str]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), duration_str, ha="center", va="bottom", fontsize=12, color="black")

plt.tight_layout()
plt.show()


# In[ ]:


test_error_rate = error_sessions_test / total_sessions_test
control_error_rate = error_sessions_control / total_sessions_control

groups = ["test", "control"]
error_rates = [test_error_rate, control_error_rate]
colors = ["darkorange", "royalblue"]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(groups, error_rates, color=colors, width=0.4)

ax.set_ylabel("error rate")
ax.set_title("error rate comparison between test and control groups")

for bar, rate in zip(bars, error_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, rate, f"{rate:.2%}", ha="center", va="bottom", fontsize=12, color="black")

plt.tight_layout()
plt.show()


# # 4. Hypotheses testing

# ## 4.1. completion rate
# Test group had a higher completion rate compared to the Control group.\
# H0: This difference is not statistically significant.\
# H1: The difference is statistically significant.

# In[ ]:


# Completion rates and number of observations
x1, x2 = confirm_count_test, confirm_count_control
n1, n2 = unique_visit_count_test, unique_visit_count_control

# Part 1: Two-sided test for completion rate comparison
stat, pval = proportions_ztest([x1, x2], [n1, n2], alternative="two-sided")
print("Part 1 - Completion Rate Comparison")
print(f"Z-statistic: {stat:.2f}, P-value: {pval:.4f}")

# Interpretation
alpha = 0.05
if pval < alpha:
    print("Reject H0: Significant difference in completion rates between the Test and Control groups.")
else:
    print("Fail to reject H0: No significant difference in completion rates.")

# Part 2: One-sided test for comparing against Control + 5%
# Adjusted completion rate for Control by adding 5%
control_completion_rate_adjusted = (x2 + 0.05 * n2) / n2

# Comparing Test completion rate directly to adjusted Control rate
print("\nPart 2 - Completion Rate with Cost-Effectiveness Threshold")
completion_rate_test = x1 / n1
if completion_rate_test > control_completion_rate_adjusted and pval < alpha:
    print(f"Reject H0: Test group's completion rate exceeds Control's by >5%, indicating cost-effectiveness.")
else:
    print("Fail to reject H0: Test group's completion rate does not exceed Control's by >5%, indicating lack of cost-effectiveness.")


# # 4.2. error rate
# Test group had a higher error rate 26,8% compared to the Control group 20.22%.\
# H0: This difference is not statistically significant.\
# H1: The difference is statistically significant.

# In[ ]:


# Calculate error rates and number of observations
x1, x2 = error_sessions_test, error_sessions_control
n1, n2 = total_sessions_test, total_sessions_control

# Two-sided test for error rate comparison
stat, pval = proportions_ztest([x1, x2], [n1, n2], alternative='two-sided')
print("Error Rate Comparison Between Test and Control Groups")
print(f"Z-statistic: {stat:.2f}, P-value: {pval:.4f}")

# Interpretation
alpha = 0.05
if pval < alpha:
    print("Reject H0: Significant difference in error rates between the Test and Control groups.")
else:
    print("Fail to reject H0: No significant difference in error rates.")


# ## 4.3. t-test of various means between the groups

# In[ ]:


def compare_groups(df, column_name, group_col="experiment_group", test_label="Test", control_label="Control"):
    # Data filtering
    test_group = df[df[group_col] == test_label][column_name]
    control_group = df[df[group_col] == control_label][column_name]
    
    # Calculate means
    test_mean = test_group.mean()
    control_mean = control_group.mean()
    
    # Perform t-test
    stat, p = ttest_ind(test_group, control_group)
    
    # Print results
    print(f"\nComparison based on {column_name}:")
    print(f"The average {column_name} of the Test users: {test_mean:.3f}")
    print(f"The average {column_name} of the Control users: {control_mean:.3f}")
    print(f"{column_name.capitalize()} comparison between groups - T-statistic: {stat:.3f}, P-value: {p:.3f}")
    
    # Interpretation
    if p < 0.05:
        print(f"With the P-value {p:.3f}, we reject the H0.")
        print(f"The average {column_name} of Test group users is significantly different than that of Control group.\n")
    else:
        print(f"With the P-value {p:.3f}, we fail to reject the H0.")
        print(f"The average {column_name} of Test group users is not significantly different than that of Control group.\n")

# Example usage
compare_groups(df_client_profiles, "client_age")
compare_groups(df_client_profiles, "client_tenure_in_years")
compare_groups(df_client_profiles, "number_of_accounts")
compare_groups(df_client_profiles, "balance")
compare_groups(df_client_profiles, "calls_per_year")
compare_groups(df_client_profiles, "logons_per_year")


# # 5. Experiment evaluation

# In[ ]:


# There's 23% difference in size between the experiment groups, which seems a bit off-balance.
# The experiment duration of 97 days seems sufficient.

print(f"Test group's number of visits: {unique_visit_count_test}")
print(f"Control group's number of visits: {unique_visit_count_control}")
balance = abs(unique_visit_count_test - unique_visit_count_control) / ((unique_visit_count_test + unique_visit_count_control) / 2)
print(f"Balance between groups: {balance:.2f}\n")

experiment_start = df_digital_footprint['date_time'].min()
experiment_end = df_digital_footprint['date_time'].max()
experiment_duration = experiment_end - experiment_start

print(f"Experiment duration: {experiment_duration.days} days")


# In[ ]:


df_digital_footprint.to_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/final_for_tableau/df_digital_footprint_final_for_tableau.csv", index=False)

