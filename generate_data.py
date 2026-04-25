# The script has 4 sections:

# 1. Setup — libraries, parameters, department sizes
# 2. Define Structure - Generate employee profiles — one row per employee
# 3. Generate Data - Generate yearly data — 5 rows per employee with realistic trends
# 4. Save Output - Save to CSV

# What do I need? → imports and setup
# What am I building? → data structures
# How do I build it? → main logic
# What do I do with it? → output

# Section 1: Setup
# Import libraries
import pandas as pd   # for creating and saving dataframes (tables)
import numpy as np    # for random number generation and math

# Set random seed — ensures same dataset every time script is run
np.random.seed(42)

# Total number of employees to generate
N_EMPLOYEES = 5000

# 5 years of data per employee
YEARS = [2021, 2022, 2023, 2024, 2025]

# List of departments in NovaTech Solutions
DEPARTMENTS = ['HR', 'Legal', 'Sales', 'Marketing', 'Technology', 'Finance', 'Retail']

# List of role levels — from most senior to most junior
ROLE_LEVELS = ['Executive', 'Manager', 'Senior', 'Individual Contributor']

# Department size as a proportion of total headcount
# Must add up to 1.0 (100%)
# Technology is largest at 28%, Legal smallest at 4%
DEPT_SIZES = {
    'Technology': 0.28,
    'Sales': 0.22,
    'Retail': 0.18,
    'Marketing': 0.12,
    'Finance': 0.10,
    'HR': 0.06,
    'Legal': 0.04
}

# Section 2: Generate employee profiles
def generate_employees():
    employees = []

    # Loop 5,000 times — one iteration per employee
    for i in range(N_EMPLOYEES):

        # Pick a department using weighted probabilities
        # Technology picked 28% of the time, Legal 4% etc.
        dept = np.random.choice(
            list(DEPT_SIZES.keys()),
            p=list(DEPT_SIZES.values())
        )

        # Pick a role level using weighted probabilities
        # 5% Executive, 20% Manager, 35% Senior, 40% Individual Contributor
        role = np.random.choice(
            ROLE_LEVELS,
            p=[0.05, 0.20, 0.35, 0.40]
        )

        # Build one employee as a dictionary
        # Each key becomes a column, each value becomes that employee's data
        employee = {
            'employee_id': f'EMP-{i+1:04d}', # EMP-0001, EMP-0002 etc.
            'department': dept,                         # assigned department
            'role_level': role,                         # assigned role level
            'age': np.random.randint(22, 58),           # random age between 22 and 57
            'gender': np.random.choice(['M', 'F'], p=[0.52, 0.48]),  # 52% male, 48% female
            'salary_band': np.random.randint(1,6)  # salary band 1 (lowest) to 5 (highest)
        }

        # Add this employee to the list
        employees.append(employee)

    # Convert list of dictionaries to a pandas DataFrame
    return pd.DataFrame(employees)

# Three types of employees we're engineering:
# Type 1 — Leavers (attrition = Yes)

# Engagement score declines year on year
# Engagement activity drops steadily
# Online and F2F learning spikes in final 1-2 years
# Absenteeism increases
# Overtime increases
# These are the patterns the model learns to detect

# Type 2 — At risk (attrition = No but high risk)

# Similar patterns to leavers but less severe
# Early warning signals present

# Type 3 — Stable (attrition = No)

# Engagement stable or improving
# Learning steady, no spike
# Normal absenteeism and overtime

# Section 3: Generate yearly data per employee
def generate_yearly_data(employees_df):

    yearly_data = []

    for _, emp in employees_df.iterrows():

        # Assign employee type
        # 18.7% will leave — split into clear and uncertain leavers
        # 81.3% will stay — split into stable and at-risk stayers
        rand = np.random.random()

        if rand < 0.075:
            emp_type = 'clear_leaver'       # 7.5% — strong signals, model very confident
            will_leave = 1
        elif rand < 0.187:
            emp_type = 'uncertain_leaver'   # 11.2% — weak signals, model uncertain
            will_leave = 1
        elif rand < 0.387:
            emp_type = 'at_risk_stayer'     # 20% — shows some leaver signals but stays
            will_leave = 0
        else:
            emp_type = 'stable_stayer'      # 61.3% — clear low risk
            will_leave = 0
        
        # Step 2 — assign will_leave based on emp_type
        if emp_type == 'clear_leaver':
            will_leave = 1 if np.random.random() < 0.85 else 0
        elif emp_type == 'uncertain_leaver':
            will_leave = 1 if np.random.random() < 0.55 else 0
        elif emp_type == 'at_risk_stayer':
            will_leave = 1 if np.random.random() < 0.20 else 0
        else:
            will_leave = 1 if np.random.random() < 0.05 else 0

        # Base values per employee type
        if emp_type == 'clear_leaver':
            base_engagement = np.random.uniform(3.0, 6.0)
            base_engagement_activity = np.random.uniform(30, 65)
            base_online_learning = np.random.uniform(3.0, 6.0)
            base_f2f_learning = np.random.uniform(2.0, 4.0)
            base_absenteeism = np.random.uniform(3.0, 6.0)
            base_overtime = np.random.uniform(8.0, 14.0)
            base_manager_score = np.random.uniform(2.0, 5.0)
            base_performance = np.random.uniform(3.0, 6.0)
            base_months_promotion = np.random.randint(24, 48)

        elif emp_type == 'uncertain_leaver':
            base_engagement = np.random.uniform(5.0, 7.5)
            base_engagement_activity = np.random.uniform(50, 80)
            base_online_learning = np.random.uniform(2.0, 5.0)
            base_f2f_learning = np.random.uniform(1.0, 3.5)
            base_absenteeism = np.random.uniform(1.5, 4.0)
            base_overtime = np.random.uniform(5.0, 10.0)
            base_manager_score = np.random.uniform(4.0, 7.0)
            base_performance = np.random.uniform(4.0, 7.5)
            base_months_promotion = np.random.randint(12, 36)

        elif emp_type == 'at_risk_stayer':
            base_engagement = np.random.uniform(4.5, 7.0)
            base_engagement_activity = np.random.uniform(45, 75)
            base_online_learning = np.random.uniform(2.0, 5.0)
            base_f2f_learning = np.random.uniform(1.0, 3.5)
            base_absenteeism = np.random.uniform(1.5, 4.5)
            base_overtime = np.random.uniform(5.0, 11.0)
            base_manager_score = np.random.uniform(3.5, 6.5)
            base_performance = np.random.uniform(4.0, 7.0)
            base_months_promotion = np.random.randint(12, 36)

        else:  # stable_stayer
            base_engagement = np.random.uniform(7.0, 9.5)
            base_engagement_activity = np.random.uniform(70, 98)
            base_online_learning = np.random.uniform(1.5, 4.0)
            base_f2f_learning = np.random.uniform(1.0, 3.0)
            base_absenteeism = np.random.uniform(0.5, 2.5)
            base_overtime = np.random.uniform(2.0, 7.0)
            base_manager_score = np.random.uniform(6.5, 9.5)
            base_performance = np.random.uniform(6.0, 10.0)
            base_months_promotion = np.random.randint(6, 24)

        # Generate one row per year
        for year_idx, year in enumerate(YEARS):

            # --- ENGAGEMENT SCORE ---
            if emp_type == 'clear_leaver':
                # Steady accelerating decline
                decline = year_idx * np.random.uniform(0.5, 0.9)
                engagement = max(1.0, base_engagement - decline + np.random.uniform(-0.5, 0.5))
            elif emp_type == 'uncertain_leaver':
                # Mild decline with noise
                decline = year_idx * np.random.uniform(0.1, 0.4)
                engagement = max(2.0, base_engagement - decline + np.random.uniform(-0.9, 0.9))
            elif emp_type == 'at_risk_stayer':
                # Fluctuates — sometimes looks like leaver, sometimes not
                engagement = max(2.0, base_engagement + np.random.uniform(-1.2, 0.8))
            else:
                # Stable with small fluctuations
                engagement = min(10.0, base_engagement + np.random.uniform(-0.6, 0.6))

            # --- ENGAGEMENT ACTIVITY ---
            if emp_type == 'clear_leaver':
                decline = year_idx * np.random.uniform(6, 12)
                engagement_activity = max(10, base_engagement_activity - decline + np.random.uniform(-5, 5))
            elif emp_type == 'uncertain_leaver':
                decline = year_idx * np.random.uniform(2, 6)
                engagement_activity = max(20, base_engagement_activity - decline + np.random.uniform(-8, 8))
            elif emp_type == 'at_risk_stayer':
                engagement_activity = max(20, base_engagement_activity + np.random.uniform(-10, 8))
            else:
                engagement_activity = min(100, base_engagement_activity + np.random.uniform(-6, 6))

            # --- ONLINE LEARNING ---
            if emp_type == 'clear_leaver' and year_idx >= 3:
                # Strong spike in final 2 years
                online_learning = base_online_learning * np.random.uniform(2.0, 3.5)
            elif emp_type == 'uncertain_leaver' and year_idx >= 3:
                # Mild spike — not always
                if np.random.random() < 0.6:
                    online_learning = base_online_learning * np.random.uniform(1.3, 2.0)
                else:
                    online_learning = base_online_learning + np.random.uniform(-0.5, 0.5)
            elif emp_type == 'at_risk_stayer' and year_idx >= 3:
                # Occasional spike — false signal
                if np.random.random() < 0.3:
                    online_learning = base_online_learning * np.random.uniform(1.2, 1.8)
                else:
                    online_learning = base_online_learning + np.random.uniform(-0.5, 0.5)
            else:
                online_learning = base_online_learning + np.random.uniform(-0.5, 0.5)

            # --- F2F LEARNING ---
            if emp_type == 'clear_leaver' and year_idx >= 3:
                f2f_learning = base_f2f_learning * np.random.uniform(1.8, 3.0)
            elif emp_type == 'uncertain_leaver' and year_idx >= 3:
                if np.random.random() < 0.5:
                    f2f_learning = base_f2f_learning * np.random.uniform(1.2, 1.8)
                else:
                    f2f_learning = base_f2f_learning + np.random.uniform(-0.4, 0.4)
            else:
                f2f_learning = base_f2f_learning + np.random.uniform(-0.4, 0.4)

            # --- ABSENTEEISM ---
            if emp_type == 'clear_leaver':
                absenteeism = base_absenteeism + (year_idx * np.random.uniform(0.5, 1.2))
            elif emp_type == 'uncertain_leaver':
                absenteeism = base_absenteeism + (year_idx * np.random.uniform(0.2, 0.6))
            elif emp_type == 'at_risk_stayer':
                absenteeism = base_absenteeism + np.random.uniform(-0.5, 1.0)
            else:
                absenteeism = base_absenteeism + np.random.uniform(-0.5, 0.5)

            # --- OVERTIME HOURS ---
            if emp_type == 'clear_leaver':
                overtime = base_overtime + (year_idx * np.random.uniform(0.8, 1.8))
            elif emp_type == 'uncertain_leaver':
                overtime = base_overtime + (year_idx * np.random.uniform(0.3, 0.9))
            elif emp_type == 'at_risk_stayer':
                overtime = base_overtime + np.random.uniform(-1.0, 2.0)
            else:
                overtime = base_overtime + np.random.uniform(-1.0, 1.0)

            # --- MANAGER SCORE ---
            # Add small year-on-year variation
            manager_score = max(1.0, min(10.0, base_manager_score + np.random.uniform(-0.8, 0.8)))

            # --- PERFORMANCE RATING ---
            performance_rating = max(1.0, min(10.0, base_performance + np.random.uniform(-0.8, 0.8)))

            # --- TRAINING HOURS ---
            training_hours = np.random.uniform(10, 80)

            # --- MONTHS SINCE LAST PROMOTION ---
            months_since_promotion = max(1, base_months_promotion + (year_idx * np.random.randint(0, 4)))

            # --- TENURE ---
            base_tenure = np.random.uniform(0.5, 15.0)
            tenure = round(base_tenure + (year_idx * 1.0), 1)

            # Build row
            row = {
                'employee_id': emp['employee_id'],
                'year': year,
                'department': emp['department'],
                'role_level': emp['role_level'],
                'age': emp['age'] + year_idx,
                'gender': emp['gender'],
                'salary_band': emp['salary_band'],
                'tenure': tenure,
                'engagement_score': round(engagement, 2),
                'engagement_activity': round(engagement_activity, 1),
                'online_learning': round(online_learning, 1),
                'f2f_learning': round(f2f_learning, 1),
                'absenteeism': round(absenteeism, 1),
                'overtime_hours': round(overtime, 1),
                'manager_score': round(manager_score, 2),
                'performance_rating': round(performance_rating, 2),
                'training_hours': round(training_hours, 1),
                'months_since_promotion': months_since_promotion,
                'attrition': will_leave
            }

            yearly_data.append(row)

    return pd.DataFrame(yearly_data)  

              
# Section 4: Run and save
def main():

    print("Generating employee profiles...")
    employees_df = generate_employees()
    print(f"Created {len(employees_df)} employee profiles")

    print("Generating yearly data...")
    data_df = generate_yearly_data(employees_df)
    print(f"Created {len(data_df)} rows of yearly data")

    # Save to CSV
    data_df.to_csv('employee_data.csv', index=False)
    print("Saved to employee_data.csv")

    # Quick summary
    print("\n--- Summary ---")
    print(f"Total rows: {len(data_df)}")
    print(f"Employees: {data_df['employee_id'].nunique()}")
    print(f"Years: {data_df['year'].unique()}")
    print(f"Attrition rate: {data_df['attrition'].mean():.1%}")
    print(f"Departments: {data_df['department'].value_counts().to_dict()}")


# Run the script
if __name__ == "__main__":
    main()